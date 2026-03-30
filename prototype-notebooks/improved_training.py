#!/usr/bin/env python3

# In this file I am implementing the fixes that I came up with
# (But for the Normal CNN learns image feature - not residual signals)

"""
improved_training.py
====================
Fixes the 53% test accuracy gap vs 80%+ train/val in genai_classifier_images.ipynb.

Key changes:
  1. Stronger augmentation  : ColorJitter + RandomRotation + GaussianBlur
  2. Two-phase fine-tuning  : 3-epoch warm-up (head only), then unfreeze layer3+layer4
  3. Discriminative LRs     : head=1e-3, backbone=1e-5  (prevents catastrophic forgetting)
  4. Dropout(0.4) in head   : reduces overfitting to training-set artefacts
  5. 25 epochs + early stop : patience=5, monitors val_loss

Run from the project root (same directory as requirements.txt):
    python prototype-notebooks/improved_training.py
Or from inside prototype-notebooks/:
    python improved_training.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Paths (same as the notebook) ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "requirements.txt").exists():
    # fallback: script is run from inside prototype-notebooks/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = PROJECT_ROOT / "output"
TRAIN_PATH  = str(OUTPUT_ROOT / "train_balanced")
VAL_PATH    = str(OUTPUT_ROOT / "validation_balanced")
TEST_PATH   = str(OUTPUT_ROOT / "test_balanced")

CHECKPOINT_PATH = str(Path(__file__).resolve().parent / "genai_resnet50_v2_best.pth")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
BATCH_SIZE    = 16
NUM_WORKERS   = 0
WARM_UP_EPOCHS = 3       # epochs where only the head trains
N_EPOCHS      = 25       # max total epochs
PATIENCE      = 5        # early stopping patience
HEAD_LR       = 1e-3     # learning rate for the classification head
BACKBONE_LR   = 1e-5     # learning rate for unfrozen backbone blocks (100× smaller)
WEIGHT_DECAY  = 1e-4

# ── FIX 1: Stronger augmentation ─────────────────────────────────────────────
# Why: training and test sets differ in style/texture; these augmentations
# force the model to learn features that survive such variation.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    # Colour shifts – AI-gen images often have distinctive colour distributions
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    # Small rotation – the classifier should not rely on exact orientation
    transforms.RandomRotation(degrees=15),
    # Slight blur – prevents over-reliance on high-frequency artefacts in train set
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Datasets & loaders ────────────────────────────────────────────────────────
train_dataset = ImageFolder(root=TRAIN_PATH, transform=train_transform)
val_dataset   = ImageFolder(root=VAL_PATH,   transform=val_test_transform)
test_dataset  = ImageFolder(root=TEST_PATH,  transform=val_test_transform)

print(f"Classes : {train_dataset.classes}")
print(f"Train   : {len(train_dataset)}")
print(f"Val     : {len(val_dataset)}")
print(f"Test    : {len(test_dataset)}")

trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
valloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
testloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ── FIX 2 + 4: Model with Dropout head  ───────────────────────────────────────
# We still start frozen (same as before) then unfreeze layer3+layer4 after warm-up.
class AiGenModel(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # FIX 4: replace the FC with Dropout + Linear for regularisation
        in_features = backbone.fc.in_features   # 2048 for ResNet50
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),                  # prevents head from memorising train artefacts
            nn.Linear(in_features, num_classes),
        )
        self.feature_extractor = backbone

        if freeze_backbone:
            # Freeze everything except our new FC head
            for name, param in self.feature_extractor.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)


num_classes = len(train_dataset.classes)
net = AiGenModel(num_classes=num_classes, freeze_backbone=True).to(device)

# ── Helper: count trainable params ───────────────────────────────────────────
def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nPhase 1 – warm-up: trainable params = {count_trainable(net):,}")

# ── FIX 2: Two-phase fine-tuning helpers ─────────────────────────────────────
def unfreeze_later_blocks(model):
    """Unfreeze layer3 and layer4 of the ResNet backbone for fine-tuning.

    We keep layer1 and layer2 frozen because low-level features (edges,
    textures) learned from ImageNet are still useful and do not need updating.
    """
    for name, param in model.feature_extractor.named_parameters():
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True


def make_optimizer(model):
    """Build an AdamW optimizer with discriminative learning rates.

    Head params get HEAD_LR; newly unfrozen backbone params get BACKBONE_LR
    (100× smaller) to avoid catastrophic forgetting of ImageNet features.
    """
    head_params     = [p for n, p in model.feature_extractor.named_parameters()
                       if "fc" in n and p.requires_grad]
    backbone_params = [p for n, p in model.feature_extractor.named_parameters()
                       if "fc" not in n and p.requires_grad]
    groups = [{"params": head_params, "lr": HEAD_LR}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": BACKBONE_LR})
    return optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


# ── Loss & initial optimizer (warm-up: head only) ────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = make_optimizer(net)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# ── FIX 5: Training loop (25 epochs + early stopping) ────────────────────────
best_val_loss = float("inf")
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
fine_tune_started = False

for epoch in range(N_EPOCHS):
    # ── Phase transition: unfreeze after warm-up ─────────────────────────────
    if epoch == WARM_UP_EPOCHS and not fine_tune_started:
        print(f"\n{'='*60}")
        print(f"Warm-up done. Unfreezing layer3 + layer4 for fine-tuning.")
        unfreeze_later_blocks(net)
        optimizer = make_optimizer(net)  # rebuild with discriminative LRs
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        fine_tune_started = True
        print(f"Phase 2 – fine-tune: trainable params = {count_trainable(net):,}")
        print(f"{'='*60}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    epoch_start = time.perf_counter()
    net.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc  = correct / total

    # ── Validate ─────────────────────────────────────────────────────────────
    net.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_dataset)
    val_acc  = correct / total

    scheduler.step(val_loss)
    epoch_time = time.perf_counter() - epoch_start

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    phase = "WARM-UP " if epoch < WARM_UP_EPOCHS else "FINE-TUNE"
    print(f"[{phase}] Epoch {epoch+1:02d}/{N_EPOCHS} | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
          f"{epoch_time:.1f}s")

    # ── Early stopping + checkpoint ──────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(net.state_dict(), CHECKPOINT_PATH)
        print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

print("\nTraining complete.")

# ── Evaluation on test set ────────────────────────────────────────────────────
print(f"\nLoading best checkpoint from {CHECKPOINT_PATH} ...")
net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
net.eval()

class_names = train_dataset.classes
class_correct = [0] * num_classes
class_total   = [0] * num_classes
all_correct, all_total = 0, 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        all_total   += labels.size(0)
        all_correct += (predicted == labels).sum().item()

        for c in range(num_classes):
            mask = (labels == c)
            class_total[c]   += mask.sum().item()
            class_correct[c] += (predicted[mask] == labels[mask]).sum().item()

print(f"\n{'='*50}")
print(f"TEST RESULTS")
print(f"{'='*50}")
print(f"Overall accuracy : {all_correct/all_total:.4f}  ({all_correct}/{all_total})")
print()
for c in range(num_classes):
    acc = class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
    print(f"  {class_names[c]:15s} : {acc:.4f}  ({class_correct[c]}/{class_total[c]})")
print(f"{'='*50}")
print(f"\nBaseline (frozen backbone, 1 epoch): overall=0.5300, "
      f"ai_generated=0.7453, real=0.3949")
