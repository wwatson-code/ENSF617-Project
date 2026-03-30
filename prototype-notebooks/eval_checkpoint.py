#!/usr/bin/env python3

# improved_training.py is the file associated.

"""
eval_checkpoint.py
Loads genai_resnet50_v2_best.pth, sweeps decision thresholds,
and shows which threshold gives the best test accuracy.

Run: python eval_checkpoint.py
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_ROOT  = PROJECT_ROOT / "output"
TEST_PATH    = str(OUTPUT_ROOT / "test_balanced")
CKPT_PATH    = str(HERE / "genai_resnet50_v3_weighted.pth")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ImageFolder(root=TEST_PATH, transform=transform)
testloader   = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
class_names  = test_dataset.classes   # ['ai_generated', 'real']
num_classes  = len(class_names)
print(f"Test set: {len(test_dataset)} images | Classes: {class_names}")

# ── Model ─────────────────────────────────────────────────────────────────────
class AiGenModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.feature_extractor = backbone

    def forward(self, x):
        return self.feature_extractor(x)

net = AiGenModel(num_classes=num_classes)
print(f"Loading checkpoint: {CKPT_PATH}")
net.load_state_dict(torch.load(CKPT_PATH, map_location=device))
net.to(device).eval()

# ── Collect all probabilities once (expensive) ────────────────────────────────
print("\nRunning inference on test set...")
all_probs  = []   # P(ai_generated) for each sample
all_labels = []   # true class index

with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 0]   # P(ai_generated), class index 0
        all_probs.append(probs.cpu())
        all_labels.append(labels)

all_probs  = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy()

AI_IDX   = class_names.index("ai_generated")   # 0
REAL_IDX = class_names.index("real")           # 1

# ── Sweep thresholds ──────────────────────────────────────────────────────────
# threshold = the probability above which we call something "ai_generated"
# default = 0.50 (model is equally split)
# raising threshold = model must be MORE confident to call something AI-gen
#                   → fewer false positives on real images

thresholds = np.arange(0.30, 0.91, 0.05)

print(f"\n{'Threshold':>10} | {'Overall':>8} | {'ai_gen':>8} | {'real':>8}")
print("-" * 46)

best_overall = 0.0
best_thresh  = 0.5
best_row     = None

for thresh in thresholds:
    predicted = (all_probs >= thresh).astype(int)   # 1 = ai_gen, 0 = real... wait

    # class 0 = ai_generated, class 1 = real
    # all_probs = P(ai_generated)
    # predicted = 0 (ai_generated) if P(ai_gen) >= thresh else 1 (real)
    predicted = np.where(all_probs >= thresh, AI_IDX, REAL_IDX)

    overall_acc = (predicted == all_labels).mean()

    ai_mask  = (all_labels == AI_IDX)
    real_mask = (all_labels == REAL_IDX)
    ai_acc   = (predicted[ai_mask]   == all_labels[ai_mask]).mean()   if ai_mask.sum()   > 0 else 0.0
    real_acc = (predicted[real_mask] == all_labels[real_mask]).mean() if real_mask.sum() > 0 else 0.0

    marker = " ◀ default" if abs(thresh - 0.50) < 0.001 else ""
    print(f"  {thresh:.2f}     | {overall_acc:.4f}   | {ai_acc:.4f}   | {real_acc:.4f}{marker}")

    if overall_acc > best_overall:
        best_overall = overall_acc
        best_thresh  = thresh
        best_row     = (ai_acc, real_acc)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"BEST THRESHOLD: {best_thresh:.2f}")
print(f"  Overall  : {best_overall:.4f}")
print(f"  ai_gen   : {best_row[0]:.4f}")
print(f"  real     : {best_row[1]:.4f}")
print(f"\nBaseline (frozen backbone, 1 epoch, thresh=0.50): 0.5300")
print(f"Previous run (fine-tuned,  thresh=0.50):          0.5836")
print(f"{'='*50}")
