from __future__ import annotations

import csv
import json
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
HF_DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"
HF_CONFIG_NAME = "default"
ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_EXPORT_ROOT = WORKSPACE_ROOT / "ensf617" / "output"
LABEL_MAP = {0: "real", 1: "ai_generated"}
VALID_SPLITS = {"train", "validation", "test"}
MANIFEST_FIELDS = [
    "row_index",
    "image_path",
    "label_a",
    "label_a_name",
    "label_b",
    "caption",
]

# train completed 230/420
# validation completed 90/90
# test completed 95/450

# Download settings.
START_PAGE_NUM = 30
PAGE_SIZE = 100
SPLIT = "test"
EXPORT_ROOT = DEFAULT_EXPORT_ROOT


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
    (root / split_name).mkdir(parents=True, exist_ok=True)
    return metadata_dir


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(
                f"{manifest_path} has columns {reader.fieldnames}, expected {MANIFEST_FIELDS}. "
                "Delete the existing manifest or point export_root at a clean dataset folder."
            )
        return list(reader)


def write_manifest_rows(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def normalize_manifest_rows(
    rows: list[dict[str, str]],
    *,
    split_name: str,
) -> tuple[list[dict[str, str]], bool]:
    normalized_rows: list[dict[str, str]] = []
    changed = False

    for row in rows:
        normalized_row = dict(row)
        flat_path = str(Path(split_name) / Path(normalized_row["image_path"]).name)
        if normalized_row["image_path"] != flat_path:
            normalized_row["image_path"] = flat_path
            changed = True

        if not normalized_row.get("label_a_name"):
            normalized_row["label_a_name"] = LABEL_MAP[int(normalized_row["label_a"])]
            changed = True

        normalized_rows.append(normalized_row)

    return normalized_rows, changed


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


def append_manifest_rows(
    manifest_path: Path,
    rows: list[dict[str, Any]],
    *,
    split_name: str,
) -> None:
    if not rows:
        return

    existing_row_indexes: set[int] = set()
    if manifest_path.exists():
        existing_rows = read_manifest_rows(manifest_path)
        existing_rows, changed = normalize_manifest_rows(existing_rows, split_name=split_name)
        if changed:
            write_manifest_rows(manifest_path, existing_rows)
        existing_row_indexes = {int(row["row_index"]) for row in existing_rows}

    rows_to_write = [row for row in rows if int(row["row_index"]) not in existing_row_indexes]
    if not rows_to_write:
        return

    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows_to_write)


def download_image(image_url: str, image_path: Path) -> None:
    if image_path.exists():
        return

    with urlopen(image_url) as response:
        image = Image.open(BytesIO(response.read())).convert("RGB")
        image.save(image_path)


def download_page(
    *,
    split: str,
    page_num: int,
    page_size: int,
    export_root: Path,
) -> tuple[int, int]:
    split_name = normalize_split(split)
    metadata_dir = prepare_export_dirs(export_root, split_name)
    offset = (page_num - 1) * page_size
    batch = fetch_rows_batch(split_name, offset=offset, length=page_size)
    manifest_path = metadata_dir / f"{split_name}_manifest.csv"
    manifest_rows: list[dict[str, Any]] = []

    for row_entry in batch["rows"]:
        row_idx = int(row_entry["row_idx"])
        row = row_entry["row"]
        label_a_value = int(row["Label_A"])
        label_b_value = int(row["Label_B"])
        label_name = LABEL_MAP[label_a_value]
        relative_path = Path(split_name) / f"{split_name}_{row_idx:06d}.png"
        image_path = export_root / relative_path

        download_image(row["Image"]["src"], image_path)

        manifest_rows.append(
            {
                "row_index": row_idx,
                "image_path": str(relative_path),
                "label_a": label_a_value,
                "label_a_name": label_name,
                "label_b": label_b_value,
                "caption": row["Caption"],
            }
        )

    append_manifest_rows(manifest_path, manifest_rows, split_name=split_name)
    returned_rows = len(batch["rows"])
    total_rows = int(batch["num_rows_total"])
    return returned_rows, total_rows


def download_data(
    page_num: int,
    page_size: int,
    split: str,
    export_root: Path | str = DEFAULT_EXPORT_ROOT,
) -> None:
    if page_num < 1:
        raise ValueError("page_num must be >= 1")
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size must be between 1 and 100")

    split_name = normalize_split(split)
    root = Path(export_root)

    current_page = page_num
    returned_rows, total_rows = download_page(
        split=split_name,
        page_num=current_page,
        page_size=page_size,
        export_root=root,
    )
    total_pages = ceil(total_rows / page_size)
    print(
        f"Downloaded page {current_page}/{total_pages} for {split_name}: "
        f"{returned_rows} rows"
    )

    while returned_rows > 0 and current_page < total_pages:
        current_page += 1
        returned_rows, _ = download_page(
            split=split_name,
            page_num=current_page,
            page_size=page_size,
            export_root=root,
        )
        print(
            f"Downloaded page {current_page}/{total_pages} for {split_name}: "
            f"{returned_rows} rows"
        )


def main() -> None:
    download_data(
        page_num=START_PAGE_NUM,
        page_size=PAGE_SIZE,
        split=SPLIT,
        export_root=EXPORT_ROOT,
    )


if __name__ == "__main__":
    main()
