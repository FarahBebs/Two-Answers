"""Local Two Answers demo.

Each image is scored twice: once for the specific dataset category, and once
for a coarser question (tumor or not, cancer-related or not). A tumor category
is not reported as cancer. This is not a diagnosis.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from cancerai import (
    IMAGE_TASKS,
    ROOT,
    TASK_TITLES,
    category_meaning,
    load_for_inference,
    predict_pil,
    to_pil,
)

DISCLAIMER = (
    "**Not a medical device.** Two Answers is a student project. A tumor is not the same thing as cancer: "
    "meningioma, pituitary tumor, fibroadenoma, and a benign lung finding are not cancer labels. "
    "The page does not diagnose cancer, it does not estimate a person's risk, and it must not be used "
    "to make a health decision."
)
QUESTION = {
    "brain": "Tumor or no tumor",
    "breast": "Benign or malignant",
    "skin": "Cancer-related or not",
    "lung": "Malignant or not",
}
NOTES = {
    "brain": (
        "This model expects a brain MRI slice like the ones in the training set. "
        "A photo of a person, a document, or a picture of a screen is not an MRI."
    ),
    "breast": (
        "This model expects a microscope image of breast tumor tissue, like BreakHis. "
        "A photo of skin, a scan of a report, or a picture of a screen is not a histology slide."
    ),
    "skin": (
        "This model expects a dermoscopic lesion image (a magnified clinical photograph) "
        "like HAM10000. An ordinary phone photo has different lighting and scale, so treat "
        "that result as a stress test, not a skin check."
    ),
    "lung": (
        "This model expects a chest CT slice like the IQ-OTH/NCCD images. "
        "A photo of a person or a picture of a screen is not a CT slice. "
        "Benign and normal are not called cancer."
    ),
}
CACHE = {}


def model_path(task: str) -> Path:
    return ROOT / "models" / f"{task}_best.pt"


def get_model(task: str):
    if task not in CACHE:
        path = model_path(task)
        if not path.exists():
            return None
        CACHE[task] = load_for_inference(path)
    return CACHE[task]


def binary_task(task: str) -> str:
    return f"{task}_binary"


def example_rows():
    rows = []
    for task in IMAGE_TASKS:
        folder = ROOT / "examples" / task
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jpg")):
            rows.append([TASK_TITLES[task], str(path)])
    return rows


def unsure(probs) -> bool:
    order = np.argsort(probs)[::-1]
    top = float(probs[order[0]])
    gap = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else 1.0
    return top < 0.70 or gap < 0.15


def analyze(task_title, image):
    task = next((key for key in IMAGE_TASKS if TASK_TITLES[key] == task_title), None)
    if task is None:
        return "Choose an image type.", None, None, None
    if image is None:
        return "Upload an image first.", None, None, None
    loaded = get_model(task)
    if loaded is None:
        return (
            f"No trained {task_title} model was found. From this folder, run:\n"
            f"python train.py --task {task}",
            None,
            None,
            None,
        )
    model, checkpoint, gradcam, device = loaded
    classes = checkpoint["classes"]
    picture = to_pil(image)
    try:
        result = predict_pil(
            model,
            picture,
            classes,
            device,
            gradcam,
            int(checkpoint.get("input_size", 224)),
        )
    except Exception as exc:
        return f"The model could not read this image ({type(exc).__name__}: {exc}).", None, None, None

    pred = result["pred"]
    category = classes[pred]
    meaning = category_meaning(category)
    lines = [
        "1. Specific category",
        f"{category} ({result['probs'][pred] * 100:.1f}%)",
        meaning["text"],
    ]
    if unsure(result["probs"]):
        lines.append("The category model is not sure: the highest score is modest, or another category is close.")
    lines.append("")
    lines.append(f"2. {QUESTION[task]}")

    binary_scores = None
    binary = get_model(binary_task(task))
    binary_ms = None
    if binary is None:
        lines.append(
            f"The yes-or-no model is not trained yet. From this folder, run: python train.py --task {binary_task(task)}"
        )
        lines.append(f"From the category above, the grouped answer would be: {meaning['group']}.")
    else:
        binary_model, binary_checkpoint, _binary_cam, binary_device = binary
        binary_classes = binary_checkpoint["classes"]
        try:
            binary_result = predict_pil(
                binary_model,
                picture,
                binary_classes,
                binary_device,
                None,
                int(binary_checkpoint.get("input_size", 224)),
            )
        except Exception as exc:
            lines.append(f"The yes-or-no model could not score this image ({type(exc).__name__}: {exc}).")
        else:
            binary_pred = binary_result["pred"]
            binary_name = binary_classes[binary_pred]
            binary_scores = binary_result["scores"]
            binary_ms = binary_result["seconds"] * 1000.0
            lines.append(f"{binary_name} ({binary_result['probs'][binary_pred] * 100:.1f}%)")
            lines.append("This model was trained only on that coarser question. It does not name the specific category.")
            if unsure(binary_result["probs"]):
                lines.append("The yes-or-no model is not sure: the two scores are close, or the top score is modest.")
            if binary_name == meaning["group"]:
                lines.append("The two models agree.")
            else:
                lines.append(
                    "The two models disagree. The category model and the yes-or-no model point at different answers. "
                    "Neither score is a diagnosis."
                )
    lines.extend(
        [
            "",
            "A score is how strongly the picture matches a training category.",
            "It is not a diagnosis, and it is not the probability that a person has cancer.",
            "",
            NOTES[task],
            "",
            f"Category model: ResNet18, {checkpoint.get('config_name', 'finetune')}.",
            f"Category inference time: {result['seconds'] * 1000:.1f} ms",
        ]
    )
    if binary_ms is not None:
        lines.append(f"Yes-or-no inference time: {binary_ms:.1f} ms")
    if result["overlay"] is None:
        lines.append("Grad-CAM could not be drawn for this image.")
    return "\n".join(lines), result["scores"], binary_scores, result["overlay"]


def build_demo():
    with gr.Blocks(title="Two Answers") as demo:
        gr.Markdown("# Two Answers")
        gr.Markdown(
            "Each image is scored by two models. The first names the specific dataset category. "
            "The second answers a coarser question: tumor or no tumor, benign or malignant, "
            "cancer-related or not, or malignant or not, depending on the image type. "
            "A benign tumor is not reported as cancer."
        )
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                task = gr.Radio(
                    [TASK_TITLES[name] for name in IMAGE_TASKS],
                    value=TASK_TITLES["brain"],
                    label="Image type",
                )
                image = gr.Image(type="pil", label="Image", height=320)
                button = gr.Button("Run model", variant="primary")
            with gr.Column():
                text = gr.Textbox(label="Result", lines=18)
                scores = gr.Image(label="Specific category scores", height=280)
                binary_scores = gr.Image(label="Yes-or-no scores", height=220)
                heat = gr.Image(
                    label="Grad-CAM for the category model — not a tumor outline",
                    height=280,
                )
        rows = example_rows()
        if rows:
            gr.Markdown(
                "The examples are held-out images from the public datasets, so the page can be tried "
                "before any personal photos are added."
            )
            gr.Examples(examples=rows, inputs=[task, image])
        button.click(analyze, inputs=[task, image], outputs=[text, scores, binary_scores, heat])
    return demo


if __name__ == "__main__":
    missing = [TASK_TITLES[task] for task in (*IMAGE_TASKS, "brain_binary", "skin_binary") if not model_path(task).exists()]
    if missing:
        print("Still missing:", ", ".join(missing))
        print("Train them with: python train.py")
    build_demo().launch(server_name="127.0.0.1", server_port=7860)
