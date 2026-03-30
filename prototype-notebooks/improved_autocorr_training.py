#!/usr/bin/env python3

# This file is the FFT 2D transforms approach with a 63% accuracy 
#(So improving on Will's approach with the same fixes as improved_training.py)

"""
improved_autocorr_training.py
==============================
Applies the same generalisation fixes as improved_training.py to the
residual-autocorrelation approach from genai_classifier_resnet18_residual_autocorr.ipynb.

What changed vs the original notebook:
  1. Augmentation (flip/rotation on raw image BEFORE autocorr transform)
  2. Dropout(0.4) in the classification head
  3. Class-weighted loss [1.0, 5.0] — reflects 5:1 real:ai ratio in test set
  4. 25 epochs + early stopping (patience=5)  [was 10 epochs, no early stop]

What did NOT change:
  - ResidualAutocorrelationTransform is identical (noise residual + FFT autocorr)
  - ResNet18 backbone trains from scratch (no ImageNet weights — input is 1-channel)
  - All backbone layers train from the start (no warm-up phase needed)

Run: python improved_autocorr_training.py  (~45–60 min on MPS)
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from torchvision.transforms import functional as TF

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# ── Paths (same as all other notebooks) ──────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_ROOT  = PROJECT_ROOT / "output"
TRAIN_PATH   = str(OUTPUT_ROOT / "train_balanced")
VAL_PATH     = str(OUTPUT_ROOT / "validation_balanced")
TEST_PATH    = str(OUTPUT_ROOT / "test_balanced")
SAVE_CKPT    = str(HERE / "genai_resnet18_autocorr_improved_best.pth")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMAGE_SIZE           = 224
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA       = 1.0
BATCH_SIZE           = 16
NUM_WORKERS          = 2
N_EPOCHS             = 10
PATIENCE             = 5
LR                   = 1e-3
WEIGHT_DECAY         = 1e-4

# ── Residual-autocorrelation transform (unchanged from notebook) ──────────────
class EnsureMinSize:
    def __init__(self, min_size=224):
        self.min_size = min_size

    def __call__(self, image):
        w, h = image.size
        if w >= self.min_size and h >= self.min_size:
            return image
        scale     = max(self.min_size / w, self.min_size / h)
        new_w     = max(self.min_size, round(w * scale))
        new_h     = max(self.min_size, round(h * scale))
        return TF.resize(image, (new_h, new_w))


class ResidualAutocorrelationTransform:
    """
    Converts a PIL RGB image → 1-channel autocorrelation map of noise residual.

    Pipeline:
      1. Ensure image is at least 224×224
      2. Convert to grayscale
      3. Compute noise residual = gray - Gaussian_blur(gray)
      4. 2D autocorrelation via FFT: ifft2(|FFT(residual)|²)
      5. fftshift, center-crop to 224×224, normalise to [-1, 1]
    """
    def __init__(self, image_size=224, kernel_size=5, sigma=1.0):
        self.image_size     = image_size
        self.kernel_size    = kernel_size
        self.sigma          = sigma
        self.ensure_min     = EnsureMinSize(min_size=image_size)

    def __call__(self, image):
        image       = self.ensure_min(image)
        gray        = TF.rgb_to_grayscale(image, num_output_channels=1)
        gray_tensor = TF.to_tensor(gray)

        blurred  = TF.gaussian_blur(gray_tensor,
                                    kernel_size=[self.kernel_size, self.kernel_size],
                                    sigma=[self.sigma, self.sigma])
        residual = gray_tensor - blurred
        residual = residual - residual.mean()

        residual_2d = residual.squeeze(0)
        spectrum    = torch.fft.fft2(residual_2d)
        autocorr    = torch.fft.ifft2(spectrum * torch.conj(spectrum)).real
        autocorr    = torch.fft.fftshift(autocorr, dim=(-2, -1)).unsqueeze(0)
        autocorr    = TF.center_crop(autocorr, [self.image_size, self.image_size])

        autocorr = autocorr - autocorr.amin()
        autocorr = autocorr / (autocorr.amax() + 1e-6)
        autocorr = (autocorr - 0.5) / 0.5
        return autocorr


# ── FIX 1: Dataset wrapper with augmentation BEFORE autocorr ─────────────────
class AugmentedAutocorrDataset(Dataset):
    """
    Wraps ImageFolder.  For training, applies random flip/rotation to the raw
    PIL image first, then runs the autocorrelation transform.
    For val/test no augmentation is applied (same as notebook).
    """
    def __init__(self, root, autocorr_transform, augment=False):
        self.dataset          = ImageFolder(root=root)
        self.autocorr_tf      = autocorr_transform
        self.augment          = augment
        self.classes          = self.dataset.classes
        self.class_to_idx     = self.dataset.class_to_idx
        self.samples          = self.dataset.samples
        self.targets          = self.dataset.targets

        self.raw_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.dataset.loader(path).convert("RGB")
        if self.augment:
            image = self.raw_augment(image)          # augment first (raw pixels)
        tensor = self.autocorr_tf(image)             # then convert to autocorr map
        return tensor, label


# ── Model (ResNet18, 1-channel input, trained from scratch) ──────────────────
class AiGenModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=None)   # no ImageNet weights

        orig_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        in_features   = backbone.fc.in_features
        backbone.fc   = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
        self.feature_extractor = backbone

    def forward(self, x):
        return self.feature_extractor(x)


if __name__ == "__main__":
    print(f"Device: {device}")
    
    autocorr_tf = ResidualAutocorrelationTransform(IMAGE_SIZE, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

    train_dataset = AugmentedAutocorrDataset(TRAIN_PATH, autocorr_tf, augment=True)
    val_dataset   = AugmentedAutocorrDataset(VAL_PATH,   autocorr_tf, augment=False)
    test_dataset  = AugmentedAutocorrDataset(TEST_PATH,  autocorr_tf, augment=False)
    class_names   = train_dataset.classes
    num_classes   = len(class_names)

    print(f"Classes : {class_names}")
    print(f"Train   : {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    valloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    testloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    net = AiGenModel(num_classes=num_classes).to(device)
    print(f"Trainable params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    class_weights = torch.tensor([1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: ai_gen={class_weights[0]:.1f}, real={class_weights[1]:.1f}")

    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss  = float("inf")
    epochs_no_imp  = 0

    print(f"\nTraining for up to {N_EPOCHS} epochs with early stopping (patience={PATIENCE})...")
    print("=" * 65)

    for epoch in range(N_EPOCHS):
        epoch_start = time.perf_counter()

        # Train
        net.train()
        run_loss, correct, total = 0.0, 0, 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * inputs.size(0)
            _, pred   = torch.max(outputs, 1)
            total    += labels.size(0)
            correct  += (pred == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(trainloader)} | Loss: {loss.item():.4f}", end='\r')
        print() # Newline after epoch batches

        train_loss = run_loss / len(train_dataset)
        train_acc  = correct / total

        # Validate
        net.eval()
        run_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss    = criterion(outputs, labels)
                run_loss += loss.item() * inputs.size(0)
                _, pred   = torch.max(outputs, 1)
                total    += labels.size(0)
                correct  += (pred == labels).sum().item()

        val_loss = run_loss / len(val_dataset)
        val_acc  = correct / total
        scheduler.step(val_loss)
        epoch_time = time.perf_counter() - epoch_start

        print(f"Epoch {epoch+1:02d}/{N_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_imp = 0
            torch.save(net.state_dict(), SAVE_CKPT)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_imp += 1
            print(f"  No improvement ({epochs_no_imp}/{PATIENCE})")
            if epochs_no_imp >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                break

    print("\nTraining complete. Evaluating best model on test set...")

    net.load_state_dict(torch.load(SAVE_CKPT, map_location=device))
    net.eval()

    all_probs, all_labels_list = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            probs  = torch.softmax(net(inputs), dim=1)[:, 0]  # P(ai_generated)
            all_probs.append(probs.cpu())
            all_labels_list.append(labels)

    all_probs      = torch.cat(all_probs).numpy()
    all_labels_arr = torch.cat(all_labels_list).numpy()

    AI_IDX   = class_names.index("ai_generated")
    REAL_IDX = class_names.index("real")

    print(f"\n{'Threshold':>10} | {'Overall':>8} | {'ai_gen':>8} | {'real':>8}")
    print("-" * 46)
    for thresh in np.arange(0.40, 0.96, 0.05):
        pred        = np.where(all_probs >= thresh, AI_IDX, REAL_IDX)
        overall_acc = (pred == all_labels_arr).mean()
        ai_acc      = (pred[all_labels_arr == AI_IDX]   == AI_IDX).mean()
        real_acc    = (pred[all_labels_arr == REAL_IDX] == REAL_IDX).mean()
        marker = " ◀ default" if abs(thresh - 0.50) < 0.001 else ""
        print(f"  {thresh:.2f}     | {overall_acc:.4f}   | {ai_acc:.4f}   | {real_acc:.4f}{marker}")

    print(f"\n{'='*50}")
    print("Original notebook (10 epochs, no improvements): 55.72%")
    print("ResNet50 improved (our best):                   62.86%")
    print(f"{'='*50}")
