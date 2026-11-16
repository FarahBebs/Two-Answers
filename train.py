"""Train the Two Answers classifiers and write the evaluation figures.

The test set is not used to choose a model or to stop training.
Each run is kept at the epoch with the best validation macro-F1.
The test set is scored only after that choice.

Examples
--------
python train.py
python train.py --task brain --only finetune
python train.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import torch

from cancerai import (
    BATCH_SIZE,
    DATASET_INFO,
    INPUT_SIZE,
    ROOT,
    SEED,
    TASK_TITLES,
    WEIGHT_DECAY,
    benchmark_batched,
    benchmark_latency,
    build_resnet,
    classification_metrics,
    count_params,
    describe_counts,
    export_class_examples,
    get_device,
    heldout_records,
    make_loader,
    most_confused,
    pick_five,
    prepare_for_explanation,
    prepare_task,
    run_eval,
    save_balance,
    save_case_grid,
    save_comparison,
    save_confusion,
    save_curves,
    save_gradcam_grid,
    save_per_class,
    seed_everything,
    train_model,
)

CONFIGS = {
    "finetune": {
        "freeze_backbone": False,
        "lr": 1e-4,
        "epochs": 8,
        "patience": 3,
    },
    "frozen": {
        "freeze_backbone": True,
        "lr": 1e-3,
        "epochs": 6,
        "patience": 3,
    },
}


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item) and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def environment(device) -> dict:
    versions = {}
    for package in ("torch", "torchvision", "gradio", "scikit-learn", "medmnist", "pillow", "numpy", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    hw_model = ""
    memory_gb = None
    try:
        hw_model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
        memory_gb = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 1024**3)
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hw_model": hw_model,
        "memory_gb": memory_gb,
        "device": str(device),
        "packages": versions,
    }


def fmt_seconds(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} min {secs} s ({seconds:.1f} s)"
    return f"{seconds:.1f} s"


def hyperparameters(config_name: str, config: dict, input_size: int) -> dict:
    if config["freeze_backbone"]:
        architecture = "ResNet18 pretrained on ImageNet; convolutional backbone frozen; new dropout + linear head trained"
    else:
        architecture = "ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head"
    return {
        "configuration": config_name,
        "architecture": architecture,
        "input_size": input_size,
        "batch_size": BATCH_SIZE,
        "max_epochs": config["epochs"],
        "early_stopping_patience": config["patience"],
        "early_stopping_min_delta_macro_f1": 0.001,
        "learning_rate": config["lr"],
        "weight_decay": WEIGHT_DECAY,
        "dropout": 0.3,
        "optimizer": "AdamW",
        "scheduler": "cosine decay from the initial learning rate to 1e-6",
        "loss": "class-weighted cross-entropy (weights average to 1)",
        "activation": "ReLU",
        "output": "linear logits; softmax is applied only to report scores",
        "seed": SEED,
        "selection_metric": "validation macro-F1",
        "augmentation": "resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter",
        "normalization": "ImageNet mean and standard deviation",
    }


def save_checkpoint(path, model, bundle, config_name, config, test_metrics, best_val, training_seconds, params):
    public_test = {key: value for key, value in test_metrics.items() if key != "report"}
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "classes": bundle.classes,
        "task": bundle.task,
        "config_name": config_name,
        "input_size": bundle.input_size,
        "hyperparameters": hyperparameters(config_name, config, bundle.input_size),
        "test_metrics": to_builtin(public_test),
        "best_val": to_builtin(best_val),
        "training_seconds": training_seconds,
        "total_params": params[0],
        "trainable_params": params[1],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def check_gradcam(model, loader, device) -> None:
    images, _labels = next(iter(loader))
    engine = prepare_for_explanation(model)
    try:
        heat = engine(images[:1].to(device), 0)
    finally:
        engine.close()
    print(f"gradcam check shape={heat.shape} min={heat.min():.3f} max={heat.max():.3f}", flush=True)


def run_config(bundle, config_name, device, epoch_override, max_batches, model_dir, result_dir):
    config = dict(CONFIGS[config_name])
    if epoch_override:
        config["epochs"] = epoch_override
    seed_everything(SEED)
    model = build_resnet(len(bundle.classes), freeze_backbone=config["freeze_backbone"], pretrained=True)
    total_params, trainable_params = count_params(model)
    model.to(device)
    train_loader = make_loader(bundle.train_ds, BATCH_SIZE, True, SEED, device)
    val_loader = make_loader(bundle.val_ds, BATCH_SIZE, False, SEED, device)
    test_loader = make_loader(bundle.test_ds, BATCH_SIZE, False, SEED, device)
    print(
        f"\n=== {bundle.task} / {config_name}  device={device}  "
        f"trainable {trainable_params:,} / {total_params:,} parameters ===",
        flush=True,
    )
    trained = train_model(
        model,
        train_loader,
        val_loader,
        bundle.class_weights,
        device,
        epochs=config["epochs"],
        lr=config["lr"],
        weight_decay=WEIGHT_DECAY,
        patience=config["patience"],
        freeze_backbone=config["freeze_backbone"],
        task_name=bundle.task,
        config_name=config_name,
        max_batches=max_batches,
    )
    model = trained["model"]
    if max_batches is not None:
        check_gradcam(model, val_loader, device)
    loss, y_true, y_pred, probs = run_eval(
        model,
        test_loader,
        device,
        bundle.class_weights,
        max_batches=max_batches,
        return_probs=True,
    )
    test_metrics = classification_metrics(y_true, y_pred, probs, bundle.classes)
    test_metrics["loss"] = float(loss)
    test_metrics["most_confused"] = most_confused(test_metrics["confusion_matrix"], bundle.classes)
    if max_batches is None:
        test_metrics["inference_ms_single"] = benchmark_latency(model, bundle.test_ds, device, bundle.input_size)
        test_metrics["inference_ms_batched"] = benchmark_batched(model, test_loader, device)
    else:
        test_metrics["inference_ms_single"] = None
        test_metrics["inference_ms_batched"] = None

    stem = f"{bundle.task}_{config_name}"
    save_curves(
        trained["history"],
        trained["best_epoch"],
        result_dir / f"{stem}_curves.png",
        f"{TASK_TITLES[bundle.task]} — {config_name}",
    )
    save_confusion(
        test_metrics["confusion_matrix"],
        bundle.classes,
        result_dir / f"{stem}_confusion.png",
        f"{TASK_TITLES[bundle.task]} — {config_name} — test set",
    )
    report_path = result_dir / f"{stem}_report.txt"
    report_path.write_text(test_metrics["report"], encoding="utf-8")
    save_checkpoint(
        model_dir / f"{stem}.pt",
        model,
        bundle,
        config_name,
        config,
        test_metrics,
        trained["best_val"],
        trained["training_seconds"],
        (total_params, trainable_params),
    )
    public = {
        "task": bundle.task,
        "config": config_name,
        "hyperparameters": hyperparameters(config_name, config, bundle.input_size),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_epoch": trained["best_epoch"],
        "epochs_ran": trained["epochs_ran"],
        "training_seconds": trained["training_seconds"],
        "best_val": trained["best_val"],
        "history": trained["history"],
        "test": {key: value for key, value in test_metrics.items() if key != "report"},
        "split_counts": bundle.split_counts,
    }
    (result_dir / f"{stem}.json").write_text(json.dumps(to_builtin(public), indent=2), encoding="utf-8")
    print(
        f"[{bundle.task} {config_name}] best epoch {trained['best_epoch']}  "
        f"val macro-F1 {trained['best_val'].get('macro_f1', 0):.3f}  "
        f"test acc {test_metrics['accuracy']:.3f}  test macro-F1 {test_metrics['f1_macro']:.3f}  "
        f"train time {fmt_seconds(trained['training_seconds'])}",
        flush=True,
    )
    kept = dict(public)
    kept["y_true"] = y_true
    kept["y_pred"] = y_pred
    kept["probs"] = probs
    del model
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return kept


def write_case_figures(bundle, best, model_path, result_dir, device):
    indices = pick_five(best["y_true"], best["y_pred"], best["probs"])
    title = f"{TASK_TITLES[bundle.task]} — five held-out test images (not phone photos)"
    save_case_grid(
        bundle.test_ds,
        bundle.classes,
        indices,
        best["y_true"],
        best["y_pred"],
        best["probs"],
        result_dir / f"{bundle.task}_five_heldout.png",
        title,
    )
    records = heldout_records(
        bundle.test_ds, bundle.classes, indices, best["y_true"], best["y_pred"], best["probs"]
    )
    (result_dir / f"{bundle.task}_five_heldout.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_resnet(len(bundle.classes), freeze_backbone=False, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    save_gradcam_grid(
        model,
        bundle.test_ds,
        bundle.classes,
        indices,
        best["y_true"],
        device,
        bundle.input_size,
        result_dir / f"{bundle.task}_five_gradcam.png",
        f"{TASK_TITLES[bundle.task]} — Grad-CAM on the same five held-out images",
    )
    save_per_class(
        best["test"],
        bundle.classes,
        result_dir / f"{bundle.task}_per_class.png",
        f"{TASK_TITLES[bundle.task]} — test recall and F1 ({best['config']})",
    )
    del model
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def write_assignment_fields(task_results, env, path: Path) -> None:
    device = env["device"]
    hardware = env.get("hw_model") or env["machine"]
    memory = f", {env['memory_gb']} GB RAM" if env.get("memory_gb") else ""
    env_text = (
        f"Python {env['python']}, PyTorch {env['packages']['torch']}, "
        f"torchvision {env['packages']['torchvision']}, device {device}, {hardware}{memory}"
    )
    lines = [
        "# Assignment fields",
        "",
        "Numbers below are written by train.py from the finished runs.",
        "Model choice uses validation macro-F1. Test numbers are from the restored best epoch.",
        "",
        "| Field | Response |",
        "|---|---|",
        "| Project Title | Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images |",
        "| Repository | https://github.com/FarahBebs/Deep-Learning-Homework-2 |",
        "| Problem / Task | Image classification, with one model per image type |",
        "| Deployment Technology | Gradio |",
        f"| Training Environment | {env_text} |",
        "",
    ]
    for task, info in task_results.items():
        best = info["best"]
        test = best["test"]
        hyper = best["hyperparameters"]
        dataset = DATASET_INFO[task]
        single = test.get("inference_ms_single")
        batched = test.get("inference_ms_batched")
        if single is None:
            inference = "not measured in the smoke run"
        else:
            inference = f"{single:.1f} ms per image (single image); {batched:.1f} ms per image in batches of {BATCH_SIZE}"
        auc = test.get("roc_auc_macro_ovr")
        auc_text = "n/a" if auc is None else f"{auc * 100:.2f}%"
        lines.extend(
            [
                f"## {TASK_TITLES[task]}",
                "",
                "| Field | Response |",
                "|---|---|",
                f"| Dataset | {dataset['name']} |",
                f"| Dataset URL | {dataset['url']} (also {dataset['also']}) |",
                f"| Model Architecture | {hyper['architecture']} |",
                f"| Input Size | {hyper['input_size']} × {hyper['input_size']} RGB |",
                (
                    f"| Hyperparameters | batch {hyper['batch_size']}, learning rate {hyper['learning_rate']}, "
                    f"weight decay {hyper['weight_decay']}, dropout {hyper['dropout']}, "
                    f"up to {hyper['max_epochs']} epochs, early stopping patience {hyper['early_stopping_patience']} "
                    f"on validation macro-F1 (minimum improvement {hyper['early_stopping_min_delta_macro_f1']}), "
                    f"seed {hyper['seed']}. Selected configuration: {best['config']}, "
                    f"restored epoch {best['best_epoch']} of {best['epochs_ran']} run. "
                    f"Augmentation: {hyper['augmentation']}. |"
                ),
                f"| Optimizer | {hyper['optimizer']}, {hyper['scheduler']} |",
                f"| Activation / Output Function | {hyper['activation']} inside the network. {hyper['output']}. |",
                f"| Loss Function | {hyper['loss']} |",
                (
                    "| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, "
                    "per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). "
                    "IoU and mAP are not used because this module is classification, not detection or segmentation. |"
                ),
                f"| Training Environment | {env_text} |",
                (
                    f"| Best Performance | Test accuracy {test['accuracy'] * 100:.2f}%, "
                    f"macro precision {test['precision_macro'] * 100:.2f}%, "
                    f"macro recall {test['recall_macro'] * 100:.2f}%, "
                    f"macro F1 {test['f1_macro'] * 100:.2f}%, "
                    f"weighted F1 {test['f1_weighted'] * 100:.2f}%, "
                    f"macro ROC-AUC {auc_text}. "
                    f"Validation macro-F1 at the chosen epoch: {best['best_val']['macro_f1'] * 100:.2f}%. |"
                ),
                f"| Inference Time | {inference} |",
                f"| Training Time | {fmt_seconds(best['training_seconds'])} |",
                "| Deployment Technology | Gradio |",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_task(task, config_names, device, epoch_override, max_batches, model_dir, result_dir, export_examples):
    bundle = prepare_task(task, INPUT_SIZE)
    describe_counts(bundle)
    save_balance(
        bundle.split_counts,
        result_dir / f"{task}_class_balance.png",
        f"{TASK_TITLES[task]} — training images per class",
    )
    runs = []
    for config_name in config_names:
        runs.append(run_config(bundle, config_name, device, epoch_override, max_batches, model_dir, result_dir))
    best = max(runs, key=lambda row: row["best_val"]["macro_f1"])
    save_comparison(runs, result_dir / f"{task}_comparison.png", f"{TASK_TITLES[task]} — configuration comparison")
    best_path = model_dir / f"{task}_{best['config']}.pt"
    if max_batches is None:
        shutil.copyfile(best_path, ROOT / "models" / f"{task}_best.pt")
        write_case_figures(bundle, best, best_path, result_dir, device)
        if export_examples:
            export_class_examples(bundle.test_ds, bundle.classes, best["y_true"], ROOT / "examples" / task)
    print(
        f"Selected {task} model: {best['config']} "
        f"(validation macro-F1 {best['best_val']['macro_f1']:.3f}).",
        flush=True,
    )
    return {"runs": [{key: value for key, value in run.items() if key not in {'y_true', 'y_pred', 'probs'}} for run in runs], "best": best}


TASK_ORDER = [
    "brain",
    "brain_binary",
    "breast",
    "breast_binary",
    "skin",
    "skin_binary",
    "lung",
    "lung_binary",
]


def saved_best(task: str, result_dir: Path):
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    info = summary.get(task)
    if not info:
        return None
    path = result_dir / f"{task}_{info['selected']}.json"
    if not path.exists():
        return None
    public = json.loads(path.read_text(encoding="utf-8"))
    return {"best": public, "runs": info.get("runs", [])}


def main():
    parser = argparse.ArgumentParser(description="Train Two Answers image classifiers")
    parser.add_argument(
        "--task",
        choices=[
            "brain",
            "skin",
            "breast",
            "lung",
            "both",
            "brain_binary",
            "skin_binary",
            "breast_binary",
            "lung_binary",
            "binary",
            "remaining",
        ],
        default="both",
    )
    parser.add_argument("--only", choices=["finetune", "frozen"], default="")
    parser.add_argument("--epochs", type=int, default=0, help="Override the epoch cap for every run")
    parser.add_argument("--smoke", action="store_true", help="One epoch and two batches, to test the pipeline")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}", flush=True)
    if args.task == "both":
        tasks = ["brain", "skin"]
    elif args.task == "binary":
        tasks = ["brain_binary", "breast_binary", "skin_binary", "lung_binary"]
    elif args.task == "remaining":
        tasks = ["breast", "breast_binary", "lung", "lung_binary"]
    else:
        tasks = [args.task]
    # The first two image types already showed that fine-tuning beats a frozen backbone.
    # Later models reuse that recipe unless --only says otherwise.
    finetune_only = {
        "binary",
        "remaining",
        "brain_binary",
        "breast",
        "breast_binary",
        "skin_binary",
        "lung",
        "lung_binary",
    }
    if args.only:
        config_names = [args.only]
    elif args.task in finetune_only:
        config_names = ["finetune"]
    else:
        config_names = ["finetune", "frozen"]
    max_batches = 2 if args.smoke else None
    epoch_override = 1 if args.smoke else args.epochs
    if args.smoke:
        model_dir = ROOT / "models" / "smoke"
        result_dir = ROOT / "results" / "smoke"
    else:
        model_dir = ROOT / "models"
        result_dir = ROOT / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    env = environment(device)
    (result_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    task_results = {}
    for task in tasks:
        task_results[task] = run_task(
            task,
            config_names,
            device,
            epoch_override,
            max_batches,
            model_dir,
            result_dir,
            export_examples=not args.smoke,
        )
    if not args.smoke:
        summary_path = result_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
        for task, info in task_results.items():
            summary[task] = {
                "selected": info["best"]["config"],
                "runs": info["runs"],
            }
        summary = {task: summary[task] for task in TASK_ORDER if task in summary}
        summary_path.write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")
        combined = {}
        for task in summary:
            if task in task_results:
                combined[task] = task_results[task]
            else:
                previous = saved_best(task, result_dir)
                if previous is not None:
                    combined[task] = previous
        write_assignment_fields(combined, env, result_dir / "assignment_fields.md")
        print("\nTraining finished. Launch the app with: python app.py", flush=True)


if __name__ == "__main__":
    main()
