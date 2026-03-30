#!/usr/bin/env python3

# improved_autocorr_training.py. this is the file associated

"""
eval_autocorr.py
================
Evaluates the improved residual autocorrelation model (ResNet18) on the test set.
Performs a threshold sweep to find the optimal decision boundary for the 5:1 imbalanced test set.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF
from pathlib import Path

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_ROOT  = PROJECT_ROOT / "output"
TEST_PATH    = str(OUTPUT_ROOT / "test_balanced")
CKPT_PATH    = str(HERE / "genai_resnet18_autocorr_improved_best.pth")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMAGE_SIZE           = 224
BATCH_SIZE           = 32
NUM_WORKERS          = 2

# ── Transforms & Dataset (Copy from training script) ──────────────────────────
class EnsureMinSize:
    def __init__(self, min_size=224):
        self.min_size = min_size
    def __call__(self, image):
        w, h = image.size
        if w >= self.min_size and h >= self.min_size: return image
        scale     = max(self.min_size / w, self.min_size / h)
        return TF.resize(image, (max(self.min_size, round(h * scale)), max(self.min_size, round(w * scale))))

class ResidualAutocorrelationTransform:
    def __init__(self, image_size=224, kernel_size=5, sigma=1.0):
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.ensure_min = EnsureMinSize(min_size=image_size)

    def __call__(self, image):
        image       = self.ensure_min(image)
        gray        = TF.rgb_to_grayscale(image, num_output_channels=1)
        gray_tensor = TF.to_tensor(gray)
        blurred  = TF.gaussian_blur(gray_tensor, [self.kernel_size, self.kernel_size], [self.sigma, self.sigma])
        residual = gray_tensor - blurred
        residual = residual - residual.mean()
        spectrum    = torch.fft.fft2(residual.squeeze(0))
        autocorr    = torch.fft.ifft2(spectrum * torch.conj(spectrum)).real
        autocorr    = torch.fft.fftshift(autocorr, dim=(-2, -1)).unsqueeze(0)
        autocorr    = TF.center_crop(autocorr, [self.image_size, self.image_size])
        autocorr = (autocorr - autocorr.amin()) / (autocorr.amax() - autocorr.amin() + 1e-6)
        return (autocorr - 0.5) / 0.5

class AutocorrDataset(Dataset):
    def __init__(self, root, autocorr_tf):
        self.dataset = ImageFolder(root=root)
        self.autocorr_tf = autocorr_tf
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = self.dataset.loader(path).convert("RGB")
        return self.autocorr_tf(image), label

# ── Model Architecture (Copy from training script) ────────────────────────────
class AiGenModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=None)
        orig_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                                   stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, num_classes))
        self.feature_extractor = backbone
    def forward(self, x): return self.feature_extractor(x)

def run_eval():
    print(f"Loading test data from: {TEST_PATH}")
    autocorr_tf = ResidualAutocorrelationTransform()
    test_ds = AutocorrDataset(TEST_PATH, autocorr_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    class_names = test_ds.dataset.classes
    print(f"Classes: {class_names}")

    print(f"Loading checkpoint: {CKPT_PATH}")
    model = AiGenModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    print("Inference on test set...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Class 0: ai_generated, Class 1: real
            probs = torch.softmax(outputs, dim=1)[:, 0] # Prob(ai_generated)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
            if (i+1) % 20 == 0:
                print(f"  Batch {i+1}/{len(test_loader)}", end='\r')
    print()

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    AI_IDX = class_names.index("ai_generated")
    REAL_IDX = class_names.index("real")

    print(f"\n{'Threshold':>10} | {'Overall':>8} | {'ai_gen':>8} | {'real':>8}")
    print("-" * 46)
    for thresh in np.arange(0.1, 1.0, 0.05):
        pred = np.where(all_probs >= thresh, AI_IDX, REAL_IDX)
        overall_acc = (pred == all_labels).mean()
        ai_acc = (pred[all_labels == AI_IDX] == AI_IDX).mean() if any(all_labels == AI_IDX) else 0
        real_acc = (pred[all_labels == REAL_IDX] == REAL_IDX).mean() if any(all_labels == REAL_IDX) else 0
        marker = " ◀ default" if abs(thresh - 0.50) < 0.01 else ""
        print(f"  {thresh:.2f}     | {overall_acc:.4f}   | {ai_acc:.4f}   | {real_acc:.4f}{marker}")

if __name__ == "__main__":
    run_eval()
