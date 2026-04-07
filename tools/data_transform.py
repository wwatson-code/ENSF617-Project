from __future__ import annotations

import argparse
import csv
import json
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image

HF_DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"
HF_CONFIG_NAME = "default"
ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_EXPORT_ROOT = Path("output/spectrogram_data")
LABEL_MAP = {0: "real", 1: "ai_generated"}
VALID_SPLITS = {"train", "validation", "test"}
MANIFEST_FIELDS = [
    "row_index",
    "spectrogram_path",
    "label_a",
    "label_a_name",
    "label_b",
    "caption",
    "original_height",
    "original_width",
    "saved_height",
    "saved_width",
    "channels",
]

START_PAGE_NUM = 1
PAGE_SIZE = 100
SPLIT = "train"
EXPORT_ROOT = Path("output/spectrogram_data")
TARGET_SIZE = 224


def normalize_split(split: str) -> str:
    split_name = split.strip().lower()
    if split_name == "validate":
        split_name = "validation"
    if split_name not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}. Expected one of: {sorted(VALID_SPLITS)}")
    return split_name


def prepare_export_dirs(root: Path, split_name: str) -> Path:
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for class_name in LABEL_MAP.values():
        (root / split_name / class_name).mkdir(parents=True, exist_ok=True)
    return metadata_dir


def fetch_rows_batch(split: str, offset: int, length: int) -> dict[str, Any]:
    if length <= 0:
        raise ValueError("length must be positive")
    if length > 100:
        raise ValueError("The Hugging Face rows API only allows length <= 100.")

    query = urlencode(
        {
            "dataset": HF_DATASET_NAME,
            "config": HF_CONFIG_NAME,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urlopen(f"{ROWS_API_URL}?{query}") as response:
        return json.loads(response.read().decode("utf-8"))


def append_manifest_rows(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def download_rgb_image(image_url: str) -> Image.Image:
    with urlopen(image_url) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def resize_image(image: Image.Image, target_size: int) -> Image.Image:
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    return image.resize((target_size, target_size), Image.BILINEAR)


def min_max_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array_min = float(array.min())
    array_max = float(array.max())
    if array_max - array_min < eps:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - array_min) / (array_max - array_min)).astype(np.float32)


def compute_spectrogram_features(image: Image.Image) -> dict[str, np.ndarray]:
    rgb = np.asarray(image, dtype=np.float32)
    channels_first = np.transpose(rgb, (2, 0, 1))

    fft = np.fft.fft2(channels_first, axes=(-2, -1))
    fft_shifted = np.fft.fftshift(fft, axes=(-2, -1))

    magnitude = np.abs(fft)
    magnitude_shifted = np.abs(fft_shifted)
    power_spectrum = magnitude ** 2
    power_spectrum_shifted = magnitude_shifted ** 2

    log_magnitude = np.log1p(magnitude)
    log_magnitude_shifted = np.log1p(magnitude_shifted)
    log_power_spectrum = np.log1p(power_spectrum)
    log_power_spectrum_shifted = np.log1p(power_spectrum_shifted)

    spectrogram = log_power_spectrum_shifted.astype(np.float32)
    spectrogram_normalized = np.stack(
        [min_max_normalize(channel) for channel in spectrogram],
        axis=0,
    ).astype(np.float32)

    return {
        "spectrogram": spectrogram,
        "spectrogram_normalized": spectrogram_normalized,
        "magnitude": magnitude.astype(np.float32),
        "magnitude_shifted": magnitude_shifted.astype(np.float32),
        "power_spectrum": power_spectrum.astype(np.float32),
        "power_spectrum_shifted": power_spectrum_shifted.astype(np.float32),
        "log_magnitude": log_magnitude.astype(np.float32),
        "log_magnitude_shifted": log_magnitude_shifted.astype(np.float32),
        "log_power_spectrum": log_power_spectrum.astype(np.float32),
        "log_power_spectrum_shifted": log_power_spectrum_shifted.astype(np.float32),
        "fft_real": fft.real.astype(np.float32),
        "fft_imag": fft.imag.astype(np.float32),
        "fft_shifted_real": fft_shifted.real.astype(np.float32),
        "fft_shifted_imag": fft_shifted.imag.astype(np.float32),
        "saved_shape": np.asarray(channels_first.shape, dtype=np.int32),
    }


def save_spectrogram_sample(
    image_url: str,
    spectrogram_path: Path,
    *,
    target_size: int,
) -> tuple[int, int, int, int, int]:
    if spectrogram_path.exists():
        with np.load(spectrogram_path) as data:
            if "saved_shape" in data:
                channels, saved_height, saved_width = data["saved_shape"].tolist()
            elif "original_shape" in data:
                channels, saved_height, saved_width = data["original_shape"].tolist()
            else:
                spec = data["spectrogram_normalized"] if "spectrogram_normalized" in data else data["spectrogram"]
                channels, saved_height, saved_width = spec.shape

            if "original_height" in data and "original_width" in data:
                original_height = int(data["original_height"])
                original_width = int(data["original_width"])
            else:
                original_height = int(saved_height)
                original_width = int(saved_width)

        return original_height, original_width, int(saved_height), int(saved_width), int(channels)

    image = download_rgb_image(image_url)
    original_width, original_height = image.size
    resized_image = resize_image(image, target_size=target_size)

    spectrogram_features = compute_spectrogram_features(resized_image)
    spectrogram_features["original_height"] = np.asarray(original_height, dtype=np.int32)
    spectrogram_features["original_width"] = np.asarray(original_width, dtype=np.int32)

    spectrogram_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(spectrogram_path, **spectrogram_features)

    return original_height, original_width, target_size, target_size, 3


def download_page(
    *,
    split: str,
    page_num: int,
    page_size: int,
    export_root: Path,
    target_size: int,
) -> tuple[int, int]:
    split_name = normalize_split(split)
    metadata_dir = prepare_export_dirs(export_root, split_name)
    offset = (page_num - 1) * page_size
    batch = fetch_rows_batch(split_name, offset=offset, length=page_size)
    manifest_path = metadata_dir / f"{split_name}_spectrogram_manifest.csv"
    manifest_rows: list[dict[str, Any]] = []

    for row_entry in batch["rows"]:
        row_idx = int(row_entry["row_idx"])
        row = row_entry["row"]
        label_a_value = int(row["Label_A"])
        label_b_value = int(row["Label_B"])
        label_name = LABEL_MAP[label_a_value]
        relative_path = Path(split_name) / label_name / f"{split_name}_{row_idx:06d}.npz"
        spectrogram_path = export_root / relative_path

        original_height, original_width, saved_height, saved_width, channels = save_spectrogram_sample(
            row["Image"]["src"],
            spectrogram_path,
            target_size=target_size,
        )

        manifest_rows.append(
            {
                "row_index": row_idx,
                "spectrogram_path": str(relative_path),
                "label_a": label_a_value,
                "label_a_name": label_name,
                "label_b": label_b_value,
                "caption": row["Caption"],
                "original_height": original_height,
                "original_width": original_width,
                "saved_height": saved_height,
                "saved_width": saved_width,
                "channels": channels,
            }
        )

    append_manifest_rows(manifest_path, manifest_rows)
    returned_rows = len(batch["rows"])
    total_rows = int(batch["num_rows_total"])
    return returned_rows, total_rows


def download_data(
    page_num: int,
    page_size: int,
    split: str,
    export_root: Path | str = DEFAULT_EXPORT_ROOT,
    *,
    target_size: int = TARGET_SIZE,
) -> None:
    if page_num < 1:
        raise ValueError("page_num must be >= 1")
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size must be between 1 and 100")
    if target_size < 1:
        raise ValueError("target_size must be >= 1")

    split_name = normalize_split(split)
    root = Path(export_root)

    current_page = page_num
    returned_rows, total_rows = download_page(
        split=split_name,
        page_num=current_page,
        page_size=page_size,
        export_root=root,
        target_size=target_size,
    )
    total_pages = ceil(total_rows / page_size)
    print(
        f"Downloaded page {current_page}/{total_pages} for {split_name}: "
        f"{returned_rows} rows | target_size={target_size}"
    )

    while returned_rows > 0 and current_page < total_pages:
        current_page += 1
        returned_rows, _ = download_page(
            split=split_name,
            page_num=current_page,
            page_size=page_size,
            export_root=root,
            target_size=target_size,
        )
        print(
            f"Downloaded page {current_page}/{total_pages} for {split_name}: "
            f"{returned_rows} rows | target_size={target_size}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Defactify rows and save fixed-size frequency-domain spectrogram NPZ files."
    )
    parser.add_argument("--page-num", type=int, default=START_PAGE_NUM, help="Starting page number (1-indexed).")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE, help="Rows per page, max 100.")
    parser.add_argument("--split", type=str, default=SPLIT, choices=sorted(VALID_SPLITS), help="Dataset split.")
    parser.add_argument("--export-root", type=Path, default=EXPORT_ROOT, help="Root output folder.")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE, help="Fixed saved height/width.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_data(
        page_num=args.page_num,
        page_size=args.page_size,
        split=args.split,
        export_root=args.export_root,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()
