"""Run a trained Two Answers model on photos in phone_photos/.

Put images here:

    phone_photos/brain/     brain MRI photos
    phone_photos/skin/      skin close-ups

Optional labels, one file for both folders or a file inside each folder:

    filename,true_label
    photo1.jpg,melanoma

The filename is the file name only, not a path. true_label can be a class name
from the dataset (for example "Glioma" or "melanoma"). Leave it blank if you
do not know the category.

This script does not diagnose anything. It records what the model said and,
where you supplied a label, whether that label matches.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cancerai import (
    IMAGE_EXTS,
    IMAGE_TASKS,
    ROOT,
    TASK_TITLES,
    category_meaning,
    load_for_inference,
    match_class,
    predict_pil,
    short_label,
    to_pil,
)

PHOTO_ROOT = ROOT / "phone_photos"


def read_labels(folder: Path) -> dict[str, str]:
    labels = {}
    for path in (PHOTO_ROOT / "labels.csv", folder / "labels.csv"):
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("filename") or "").strip()
                label = (row.get("true_label") or "").strip()
                if name:
                    labels[name] = label
    return labels


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith(".")
    )


def run_folder(task: str, folder: Path, device_choice: str | None) -> int:
    images = list_images(folder)
    if not images:
        print(f"No images in {folder}")
        return 0
    checkpoint_path = ROOT / "models" / f"{task}_best.pt"
    if not checkpoint_path.exists():
        print(f"Missing {checkpoint_path}. Train this model first.")
        return 0
    model, checkpoint, gradcam, device = load_for_inference(checkpoint_path)
    classes = checkpoint["classes"]
    binary_path = ROOT / "models" / f"{task}_binary_best.pt"
    binary = load_for_inference(binary_path) if binary_path.exists() else None
    labels = read_labels(folder)
    rows = []
    overlays = []
    for path in images:
        picture = to_pil(path)
        result = predict_pil(
            model,
            picture,
            classes,
            device,
            gradcam,
            int(checkpoint.get("input_size", 224)),
        )
        true_text = labels.get(path.name, "")
        true_index = match_class(true_text, classes) if true_text else None
        pred = result["pred"]
        correct = None if true_index is None else bool(true_index == pred)
        finding = ""
        finding_score = None
        if binary is not None:
            binary_model, binary_checkpoint, _cam, binary_device = binary
            binary_result = predict_pil(
                binary_model,
                picture,
                binary_checkpoint["classes"],
                binary_device,
                None,
                int(binary_checkpoint.get("input_size", 224)),
            )
            finding = binary_checkpoint["classes"][binary_result["pred"]]
            finding_score = float(binary_result["probs"][binary_result["pred"]])
        else:
            finding = category_meaning(classes[pred])["group"]
        rows.append(
            {
                "file": path.name,
                "true": classes[true_index] if true_index is not None else (true_text or ""),
                "predicted": classes[pred],
                "confidence": float(result["probs"][pred]),
                "finding": finding,
                "finding_score": finding_score,
                "correct": correct,
                "ms": result["seconds"] * 1000.0,
            }
        )
        overlays.append(result["overlay"])
        status = "unlabeled" if correct is None else ("match" if correct else "mismatch")
        finding_text = finding if finding_score is None else f"{finding} ({finding_score * 100:.1f}%)"
        print(f"{path.name}: {classes[pred]} ({result['probs'][pred] * 100:.1f}%) | {finding_text}  {status}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_table(task, rows, out_dir / f"phone_{task}.md")
    _write_grid(task, images, rows, overlays, out_dir / f"phone_{task}.png")
    graded = [row for row in rows if row["correct"] is not None]
    if graded:
        hits = sum(1 for row in graded if row["correct"])
        print(f"{task}: {hits}/{len(graded)} matched the labels you supplied.")
    else:
        print(f"{task}: {len(rows)} image(s) scored. Add true_label values in labels.csv to compare them.")
    return len(rows)


def _write_table(task: str, rows, path: Path) -> None:
    lines = [
        f"# Phone photos — {TASK_TITLES[task]}",
        "",
        "These are pictures placed in `phone_photos/` and scored after training.",
        "A match means the model's top category is the same text label that was typed in `labels.csv`.",
        "It is not a medical confirmation.",
        "",
        "| File | Supplied label | Specific category | Score | Yes-or-no answer | Compared with the supplied label | Time |",
        "|---|---|---|---:|---|---|---:|",
    ]
    for row in rows:
        if row["correct"] is None:
            compared = "no label supplied"
        elif row["correct"]:
            compared = "match"
        else:
            compared = "different"
        true_text = row["true"] or "—"
        if row["finding_score"] is None:
            finding = row["finding"] or "—"
        else:
            finding = f"{row['finding']} ({row['finding_score'] * 100:.1f}%)"
        lines.append(
            f"| {row['file']} | {true_text} | {row['predicted']} | {row['confidence'] * 100:.1f}% | {finding} | {compared} | {row['ms']:.0f} ms |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_grid(task, images, rows, overlays, path: Path) -> None:
    n = len(rows)
    cols = min(n, 5)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.1 * cols, 3.4 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for axis, image_path, row, overlay in zip(axes, images, rows, overlays):
        axis.imshow(overlay if overlay is not None else to_pil(image_path))
        if row["correct"] is None:
            color = "#1f4e79"
            mark = "no supplied label"
        elif row["correct"]:
            color = "#1b7f3a"
            mark = "matches supplied label"
        else:
            color = "#b42318"
            mark = "differs from supplied label"
        true_text = short_label(row["true"]) if row["true"] else "—"
        axis.set_title(
            f"{row['file']}\nSupplied: {true_text}\n"
            f"Model: {short_label(row['predicted'])} ({row['confidence'] * 100:.0f}%)\n{mark}",
            color=color,
            fontsize=8,
        )
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[n:]:
        axis.axis("off")
    fig.suptitle(f"{TASK_TITLES[task]} — phone photos", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Score photos in phone_photos/")
    parser.add_argument("--task", choices=[*IMAGE_TASKS, "both"], default="both")
    args = parser.parse_args()
    tasks = list(IMAGE_TASKS) if args.task == "both" else [args.task]
    total = 0
    for task in tasks:
        total += run_folder(task, PHOTO_ROOT / task, None)
    if total == 0:
        print("\nNo phone photos found.")
        print("Add images to phone_photos/brain/, breast/, skin/, or lung/, then run this script again.")
        print("Optional comparison file: phone_photos/labels.csv with columns filename,true_label")


if __name__ == "__main__":
    main()
