import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "ensf617" / "output"
METADATA_ROOT = DATA_ROOT / "metadata"
RESULTS_PATH = PROJECT_ROOT / "saved_models" / "srnet_robustness_experiments_results.json"
MIDJOURNEY_LABEL = 5

TRAIN_RESIZE_SHORT_SIDE = 320
EVAL_RESIZE_SHORT_SIDE = 256
AUTOCORR_SOURCE_CROP = 256
FINAL_MAP_SIZE = 224
RESIDUAL_CROP_SIZE = 224
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0
FLIP_PROBABILITY = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 617

RESAMPLING = getattr(Image, "Resampling", Image)


class ImageRowsDataset(Dataset):
    def __init__(self, rows, split_name, representation, training):
        self.rows = rows
        self.split_name = split_name
        self.representation = representation
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = DATA_ROOT / row["image_path"]

        if not image_path.exists():
            image_path = DATA_ROOT / self.split_name / Path(row["image_path"]).name

        with Image.open(image_path) as image:
            image = image.convert("L")
            width, height = image.size
            short_side = min(width, height)

            if self.training:
                resize_target = TRAIN_RESIZE_SHORT_SIDE
            else:
                resize_target = EVAL_RESIZE_SHORT_SIDE

            if short_side < resize_target:
                scale = resize_target / short_side
                width = round(width * scale)
                height = round(height * scale)
                image = image.resize((width, height), RESAMPLING.BILINEAR)

            if self.representation == "autocorr":
                crop_size = AUTOCORR_SOURCE_CROP
            else:
                crop_size = RESIDUAL_CROP_SIZE

            if self.training:
                max_top = max(image.height - crop_size, 0)
                max_left = max(image.width - crop_size, 0)
                top = random.randint(0, max_top) if max_top > 0 else 0
                left = random.randint(0, max_left) if max_left > 0 else 0
                image = TF.crop(image, top, left, crop_size, crop_size)

                if random.random() < FLIP_PROBABILITY:
                    image = TF.hflip(image)
            else:
                image = TF.center_crop(image, [crop_size, crop_size])

        image = TF.to_tensor(image)
        blurred = TF.gaussian_blur(image, kernel_size=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA)
        residual = image - blurred

        if self.representation == "autocorr":
            spectrum = torch.fft.fft2(residual)
            output = torch.fft.ifft2(spectrum * torch.conj(spectrum)).real
            output = torch.fft.fftshift(output, dim=(-2, -1))
            output = TF.center_crop(output, [FINAL_MAP_SIZE, FINAL_MAP_SIZE])
        else:
            output = residual

        output = output - output.mean()
        output = output / (output.std() + 1e-6)

        label = int(row["label_a"])
        return output, label


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = out + identity
        out = torch.relu(out)
        return out


class SimpleSRNet(nn.Module):
    def __init__(self, stem_channels, block_specs, dropout_probability):
        super().__init__()
        self.conv1 = nn.Conv2d(1, stem_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(stem_channels)

        blocks = []
        current_channels = stem_channels

        for out_channels, stride in block_specs:
            blocks.append(ResidualBlock(current_channels, out_channels, stride=stride))
            current_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_probability)
        self.fc = nn.Linear(current_channels, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_manifest_rows(split_name):
    rows = []
    manifest_path = METADATA_ROOT / f"{split_name}_manifest.csv"

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_a = int(row["label_a"])
            label_b = int(row["label_b"])

            if label_a == 0 or label_b == MIDJOURNEY_LABEL:
                rows.append(row)

    return rows


def deduplicate_rows(train_rows, val_rows, test_rows):
    clean_rows = {}
    seen_hashes = set()

    for split_name, rows in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
        kept_rows = []
        duplicates_removed = 0

        for row in rows:
            image_path = DATA_ROOT / row["image_path"]

            if not image_path.exists():
                image_path = DATA_ROOT / split_name / Path(row["image_path"]).name

            with image_path.open("rb") as handle:
                image_hash = hashlib.md5(handle.read()).hexdigest()

            if image_hash in seen_hashes:
                duplicates_removed += 1
                continue

            seen_hashes.add(image_hash)
            kept_rows.append(row)

        clean_rows[split_name] = kept_rows
        print(
            split_name,
            "rows after deduplication:",
            len(kept_rows),
            "| duplicates removed:",
            duplicates_removed,
        )

    return clean_rows["train"], clean_rows["validation"], clean_rows["test"]


def sample_rows_per_class(rows, per_class):
    if per_class is None:
        return rows

    real_rows = [row for row in rows if int(row["label_a"]) == 0]
    ai_rows = [row for row in rows if int(row["label_a"]) == 1]

    shuffler = random.Random(SEED)
    shuffler.shuffle(real_rows)
    shuffler.shuffle(ai_rows)

    sampled_rows = real_rows[:per_class] + ai_rows[:per_class]
    shuffler.shuffle(sampled_rows)
    return sampled_rows


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(2, 2, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            predictions = outputs.argmax(dim=1)

            loss_total += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                confusion[true_label, predicted_label] += 1

    loss_value = loss_total / total
    accuracy = correct / total

    real_recall = confusion[0, 0].item() / max(confusion[0].sum().item(), 1)
    ai_recall = confusion[1, 1].item() / max(confusion[1].sum().item(), 1)
    balanced_accuracy = (real_recall + ai_recall) / 2.0

    return {
        "loss": loss_value,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "real_recall": real_recall,
        "ai_recall": ai_recall,
        "confusion": confusion.tolist(),
    }


def run_experiment(name, config, train_rows, val_rows, test_rows, args):
    set_seed(SEED)
    device = torch.device("cpu")

    train_dataset = ImageRowsDataset(
        train_rows,
        split_name="train",
        representation=config["representation"],
        training=True,
    )
    val_dataset = ImageRowsDataset(
        val_rows,
        split_name="validation",
        representation=config["representation"],
        training=False,
    )
    test_dataset = ImageRowsDataset(
        test_rows,
        split_name="test",
        representation=config["representation"],
        training=False,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = SimpleSRNet(
        stem_channels=config["stem_channels"],
        block_specs=config["block_specs"],
        dropout_probability=config["dropout_probability"],
    ).to(device)

    if config["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=config["weight_decay"],
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=config["weight_decay"],
        )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    print()
    print("=" * 80)
    print("Experiment:", name)
    print("Representation:", config["representation"])
    print("Trainable parameters:", count_parameters(model))
    print("Epochs:", args.epochs)
    print("=" * 80)

    start_time = time.time()
    history = []

    for epoch_index in range(args.epochs):
        model.train()
        loss_total = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = loss_total / total
        train_accuracy = correct / total
        val_metrics = evaluate_model(model, val_loader, loss_fn, device)

        history.append(
            {
                "epoch": epoch_index + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_real_recall": val_metrics["real_recall"],
                "val_ai_recall": val_metrics["ai_recall"],
            }
        )

        print(
            "Epoch",
            epoch_index + 1,
            "| train acc =",
            round(train_accuracy, 4),
            "| val acc =",
            round(val_metrics["accuracy"], 4),
            "| val bal acc =",
            round(val_metrics["balanced_accuracy"], 4),
            "| val AI recall =",
            round(val_metrics["ai_recall"], 4),
        )

    final_val_metrics = evaluate_model(model, val_loader, loss_fn, device)
    final_test_metrics = evaluate_model(model, test_loader, loss_fn, device)
    duration_seconds = time.time() - start_time

    print(
        "Finished",
        name,
        "| val bal acc =",
        round(final_val_metrics["balanced_accuracy"], 4),
        "| test bal acc =",
        round(final_test_metrics["balanced_accuracy"], 4),
        "| test AI recall =",
        round(final_test_metrics["ai_recall"], 4),
        "| seconds =",
        round(duration_seconds, 1),
    )

    return {
        "experiment": name,
        "representation": config["representation"],
        "parameters": count_parameters(model),
        "epochs": args.epochs,
        "optimizer": config["optimizer"],
        "weight_decay": config["weight_decay"],
        "label_smoothing": config["label_smoothing"],
        "dropout_probability": config["dropout_probability"],
        "history": history,
        "validation": final_val_metrics,
        "test": final_test_metrics,
        "duration_seconds": duration_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--val-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)

    experiments = {
        "autocorr_base": {
            "representation": "autocorr",
            "stem_channels": 32,
            "block_specs": [(32, 1), (64, 2), (64, 1), (128, 2), (128, 1)],
            "dropout_probability": 0.0,
            "label_smoothing": 0.0,
            "weight_decay": 1e-4,
            "optimizer": "adam",
        },
        "autocorr_small": {
            "representation": "autocorr",
            "stem_channels": 16,
            "block_specs": [(16, 1), (32, 2), (32, 1), (64, 2)],
            "dropout_probability": 0.0,
            "label_smoothing": 0.0,
            "weight_decay": 1e-4,
            "optimizer": "adam",
        },
        "raw_residual_base": {
            "representation": "residual",
            "stem_channels": 32,
            "block_specs": [(32, 1), (64, 2), (64, 1), (128, 2), (128, 1)],
            "dropout_probability": 0.0,
            "label_smoothing": 0.0,
            "weight_decay": 1e-4,
            "optimizer": "adam",
        },
        "raw_residual_small": {
            "representation": "residual",
            "stem_channels": 16,
            "block_specs": [(16, 1), (32, 2), (32, 1), (64, 2)],
            "dropout_probability": 0.0,
            "label_smoothing": 0.0,
            "weight_decay": 1e-4,
            "optimizer": "adam",
        },
        "raw_residual_small_reg": {
            "representation": "residual",
            "stem_channels": 16,
            "block_specs": [(16, 1), (32, 2), (32, 1), (64, 2)],
            "dropout_probability": 0.3,
            "label_smoothing": 0.05,
            "weight_decay": 5e-4,
            "optimizer": "adamw",
        },
    }

    if args.experiments is None or len(args.experiments) == 0:
        selected_names = list(experiments.keys())
    else:
        selected_names = args.experiments

    train_rows = load_manifest_rows("train")
    val_rows = load_manifest_rows("validation")
    test_rows = load_manifest_rows("test")

    print("Filtered rows before deduplication:")
    print("train:", len(train_rows))
    print("validation:", len(val_rows))
    print("test:", len(test_rows))

    train_rows, val_rows, test_rows = deduplicate_rows(train_rows, val_rows, test_rows)

    print()
    print("Filtered rows after deduplication:")
    print("train:", len(train_rows))
    print("validation:", len(val_rows))
    print("test:", len(test_rows))

    train_rows = sample_rows_per_class(train_rows, args.train_per_class)
    val_rows = sample_rows_per_class(val_rows, args.val_per_class)
    test_rows = sample_rows_per_class(test_rows, args.test_per_class)

    print()
    print("Rows used for this run:")
    print("train:", len(train_rows))
    print("validation:", len(val_rows))
    print("test:", len(test_rows))

    results = []

    for name in selected_names:
        if name not in experiments:
            raise ValueError(f"Unknown experiment name: {name}")

        result = run_experiment(name, experiments[name], train_rows, val_rows, test_rows, args)
        results.append(result)

    results.sort(key=lambda item: item["validation"]["balanced_accuracy"], reverse=True)

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(results, indent=2))

    print()
    print("Saved experiment results to:", args.results_path)
    print()
    print("Validation ranking:")
    for item in results:
        print(
            item["experiment"],
            "| val bal acc =",
            round(item["validation"]["balanced_accuracy"], 4),
            "| test bal acc =",
            round(item["test"]["balanced_accuracy"], 4),
            "| test AI recall =",
            round(item["test"]["ai_recall"], 4),
        )


if __name__ == "__main__":
    main()
