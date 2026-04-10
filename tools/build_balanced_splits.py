from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

MANIFEST_FIELDS = [
    "row_index",
    "image_path",
    "label_a",
    "label_a_name",
    "label_b",
    "caption",
]


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_balanced_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["caption"]].append(row)

    balanced_rows: list[dict[str, str]] = []
    balanced_split = f"{split}_balanced"

    for caption in sorted(grouped_rows):
        caption_rows = grouped_rows[caption]
        real_rows = [row for row in caption_rows if row["label_a_name"] == "real"]
        ai_rows = [row for row in caption_rows if row["label_a_name"] == "ai_generated"]

        if not real_rows or not ai_rows:
            continue

        selected_ai_row = rng.choice(ai_rows)

        for row in real_rows:
            balanced_row = dict(row)
            image_name = Path(row["image_path"]).name
            balanced_row["image_path"] = str(Path(balanced_split) / "real" / image_name)
            balanced_rows.append(balanced_row)

        balanced_ai_row = dict(selected_ai_row)
        ai_image_name = Path(selected_ai_row["image_path"]).name
        balanced_ai_row["image_path"] = str(Path(balanced_split) / "ai_generated" / ai_image_name)
        balanced_rows.append(balanced_ai_row)

    return balanced_rows


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def copy_images(export_root: Path, rows: list[dict[str, str]], *, split: str) -> None:
    balanced_root = export_root / f"{split}_balanced"
    if balanced_root.exists():
        shutil.rmtree(balanced_root)

    (balanced_root / "real").mkdir(parents=True, exist_ok=True)
    (balanced_root / "ai_generated").mkdir(parents=True, exist_ok=True)

    for row in rows:
        source_path = export_root / split / row["label_a_name"] / Path(row["image_path"]).name
        target_path = export_root / row["image_path"]
        shutil.copy2(source_path, target_path)


def build_balanced_split(export_root: Path, split: str, seed: int) -> list[dict[str, str]]:
    manifest_path = export_root / "metadata" / f"{split}_manifest.csv"
    rows = load_manifest_rows(manifest_path)
    balanced_rows = build_balanced_rows(rows, split=split, seed=seed)
    copy_images(export_root, balanced_rows, split=split)
    write_manifest(
        export_root / "metadata" / f"{split}_balanced_manifest.csv",
        balanced_rows,
    )
    return balanced_rows


def main() -> None:
    export_root = Path("output")
    splits = ["train", "validation", "test"]
    seed = 42

    print(f"Starting balancing process for: {splits}...")
    for split in splits:
        balanced_rows = build_balanced_split(export_root, split, seed)
        real_count = sum(row["label_a_name"] == "real" for row in balanced_rows)
        ai_count = sum(row["label_a_name"] == "ai_generated" for row in balanced_rows)
        print(
            f"{split}_balanced: wrote {len(balanced_rows)} rows "
            f"({real_count} real, {ai_count} ai_generated)"
        )


if __name__ == "__main__":
    main()
