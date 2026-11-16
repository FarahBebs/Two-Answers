"""Two Answers data, ResNet training, evaluation, and Grad-CAM.

Educational prototype. These models score categories from public datasets.
They do not diagnose cancer.
"""

from __future__ import annotations

import os
import random
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch"))

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

INPUT_SIZE = 224
SEED = 42
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
MIN_DELTA = 0.001
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
BRAIN_URLS = [
    "https://zenodo.org/records/12735702/files/brain-tumor-mri-dataset.zip?download=1",
    "https://zenodo.org/api/records/12735702/files/brain-tumor-mri-dataset.zip/content",
]
BRAIN_ORDER = ["glioma", "meningioma", "pituitary", "notumor"]
BRAIN_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor",
}
BRAIN_ALIASES = {
    "glioma": "glioma",
    "gliomatumor": "glioma",
    "meningioma": "meningioma",
    "meningiomatumor": "meningioma",
    "pituitary": "pituitary",
    "pituitarytumor": "pituitary",
    "pituitarytumour": "pituitary",
    "notumor": "notumor",
    "notumour": "notumor",
    "healthy": "notumor",
    "normal": "notumor",
}
SHORT = {
    "Glioma": "Glioma",
    "Meningioma": "Meningioma",
    "Pituitary tumor": "Pituitary",
    "No tumor": "No tumor",
    "Tumor detected": "Tumor detected",
    "No tumor detected": "No tumor detected",
    "Cancer-related": "Cancer-related",
    "Not cancer-related": "Not cancer-related",
    "actinic keratoses": "Actinic keratoses",
    "actinic keratoses and intraepithelial carcinoma": "Actinic keratoses",
    "basal cell carcinoma": "Basal cell carcinoma",
    "benign keratosis-like lesions": "Benign keratosis",
    "dermatofibroma": "Dermatofibroma",
    "melanoma": "Melanoma",
    "melanocytic nevi": "Melanocytic nevi",
    "vascular lesions": "Vascular lesions",
    "Adenosis": "Adenosis",
    "Fibroadenoma": "Fibroadenoma",
    "Phyllodes tumor": "Phyllodes tumor",
    "Tubular adenoma": "Tubular adenoma",
    "Ductal carcinoma": "Ductal carcinoma",
    "Lobular carcinoma": "Lobular carcinoma",
    "Mucinous carcinoma": "Mucinous carcinoma",
    "Papillary carcinoma": "Papillary carcinoma",
    "Benign": "Benign",
    "Malignant": "Malignant",
    "Not malignant": "Not malignant",
    "Normal": "Normal",
}
DATASET_INFO = {
    "brain": {
        "name": "Brain Tumor MRI Dataset (Nickparvar; images from figshare, SARTAJ, and Br35H)",
        "url": "https://doi.org/10.5281/zenodo.12735702",
        "also": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        "license": "CC BY 4.0",
    },
    "skin": {
        "name": "DermaMNIST (MedMNIST v2), derived from HAM10000",
        "url": "https://doi.org/10.7910/DVN/DBW86T",
        "also": "https://medmnist.com/",
        "license": "CC BY-NC 4.0 (HAM10000)",
    },
    "brain_binary": {
        "name": (
            "Same Brain Tumor MRI images, with labels collapsed to two answers: "
            "Tumor detected (glioma, meningioma, or pituitary tumor) and No tumor detected. "
            "The source labels do not say whether a tumor is benign or malignant."
        ),
        "url": "https://doi.org/10.5281/zenodo.12735702",
        "also": "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        "license": "CC BY 4.0",
    },
    "skin_binary": {
        "name": (
            "Same DermaMNIST images, with labels collapsed to Cancer-related "
            "(melanoma, basal cell carcinoma, and actinic keratoses / intraepithelial carcinoma) "
            "or Not cancer-related (nevi, benign keratosis, dermatofibroma, vascular lesions). "
            "Actinic keratoses are precancerous, not invasive cancer; they are included because "
            "the dataset treats them as a carcinoma-spectrum category."
        ),
        "url": "https://doi.org/10.7910/DVN/DBW86T",
        "also": "https://medmnist.com/",
        "license": "CC BY-NC 4.0 (HAM10000)",
    },
    "breast": {
        "name": (
            "BreakHis breast tumor microscopy (Spanhol et al.). Eight subtypes: adenosis, fibroadenoma, "
            "phyllodes tumor, tubular adenoma, ductal carcinoma, lobular carcinoma, mucinous carcinoma, "
            "and papillary carcinoma. Patients are not split across train, validation, and test."
        ),
        "url": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
        "also": "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
        "license": "free for non-commercial research; cite Spanhol et al., IEEE TBME 2016",
    },
    "breast_binary": {
        "name": (
            "Same BreakHis images, with the eight subtypes collapsed using the dataset's own folders: "
            "Benign (adenosis, fibroadenoma, phyllodes tumor, tubular adenoma) and Malignant "
            "(ductal, lobular, mucinous, and papillary carcinoma). Phyllodes tumor is benign in this "
            "dataset's folder; phyllodes tumors can be malignant outside this file."
        ),
        "url": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
        "also": "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
        "license": "free for non-commercial research; cite Spanhol et al., IEEE TBME 2016",
    },
    "lung": {
        "name": (
            "IQ-OTH/NCCD lung CT slices (alyasriy and AL-Huseiny). Three labels: Normal, Benign, and Malignant. "
            "This is slice classification, not 3D nodule detection. LIDC-IDRI was not used: it is about 125 GB "
            "of DICOM volumes. A benign slice is not labeled as cancer."
        ),
        "url": "https://doi.org/10.17632/bhmdr45bh2.4",
        "also": "https://github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset",
        "license": "CC0, via the authors' public GitHub copy of the Mendeley release",
    },
    "lung_binary": {
        "name": (
            "Same IQ-OTH/NCCD slices, collapsed to Not malignant (normal and benign) and Malignant. "
            "A benign finding stays on the not-malignant side."
        ),
        "url": "https://doi.org/10.17632/bhmdr45bh2.4",
        "also": "https://github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset",
        "license": "CC0, via the authors' public GitHub copy of the Mendeley release",
    },
}
TASK_TITLES = {
    "brain": "Brain MRI",
    "breast": "Breast histopathology",
    "skin": "Skin lesion (dermoscopy)",
    "lung": "Lung CT",
    "brain_binary": "Brain MRI — tumor or no tumor",
    "breast_binary": "Breast histopathology — benign or malignant",
    "skin_binary": "Skin lesion — cancer-related or not",
    "lung_binary": "Lung CT — malignant or not",
}
IMAGE_TASKS = ("brain", "breast", "skin", "lung")
BINARY_CLASSES = {
    "brain": ["Tumor detected", "No tumor detected"],
    "breast": ["Benign", "Malignant"],
    "skin": ["Cancer-related", "Not cancer-related"],
    "lung": ["Not malignant", "Malignant"],
}
# What each multi-class dataset label means. A tumor category is not a cancer diagnosis.
CLASS_MEANING = {
    "Glioma": {
        "group": "Tumor detected",
        "kind": "tumor",
        "text": (
            "Glioma is a tumor category in this dataset. The images are not labeled benign or malignant, "
            "so this result is not a cancer yes or no."
        ),
    },
    "Meningioma": {
        "group": "Tumor detected",
        "kind": "tumor",
        "text": (
            "Meningioma is a tumor. Many meningiomas are benign. This dataset does not label it as cancer."
        ),
    },
    "Pituitary tumor": {
        "group": "Tumor detected",
        "kind": "tumor",
        "text": (
            "A pituitary tumor is a tumor, and it is often benign. This dataset does not label it as cancer."
        ),
    },
    "No tumor": {
        "group": "No tumor detected",
        "kind": "none",
        "text": "No tumor was the closest category. That is not a statement that a person is healthy.",
    },
    "melanoma": {
        "group": "Cancer-related",
        "kind": "malignant",
        "text": "In HAM10000, melanoma is a malignant lesion category.",
    },
    "basal cell carcinoma": {
        "group": "Cancer-related",
        "kind": "malignant",
        "text": "In HAM10000, basal cell carcinoma is a malignant lesion category.",
    },
    "actinic keratoses and intraepithelial carcinoma": {
        "group": "Cancer-related",
        "kind": "precancerous",
        "text": (
            "Actinic keratoses and intraepithelial carcinoma are a precancerous category in HAM10000. "
            "They are not the same as invasive melanoma. The yes-or-no model still counts this category as cancer-related."
        ),
    },
    "benign keratosis-like lesions": {
        "group": "Not cancer-related",
        "kind": "benign",
        "text": "Benign keratosis-like lesions are a benign category in this dataset.",
    },
    "dermatofibroma": {
        "group": "Not cancer-related",
        "kind": "benign",
        "text": "Dermatofibroma is a benign category in this dataset.",
    },
    "melanocytic nevi": {
        "group": "Not cancer-related",
        "kind": "benign",
        "text": "A melanocytic nevus is a benign category in this dataset, not a cancer.",
    },
    "vascular lesions": {
        "group": "Not cancer-related",
        "kind": "benign",
        "text": "Vascular lesions are a benign category in this dataset.",
    },
    "Adenosis": {
        "group": "Benign",
        "kind": "benign",
        "text": "Adenosis is a benign breast-tissue category in BreakHis. It is not a cancer label.",
    },
    "Fibroadenoma": {
        "group": "Benign",
        "kind": "benign",
        "text": "Fibroadenoma is a benign breast tumor in BreakHis. A benign tumor is not cancer.",
    },
    "Phyllodes tumor": {
        "group": "Benign",
        "kind": "benign",
        "text": (
            "BreakHis files this phyllodes-tumor folder under benign. Phyllodes tumors can also be malignant "
            "outside this dataset, so this label is the dataset folder, not a general rule."
        ),
    },
    "Tubular adenoma": {
        "group": "Benign",
        "kind": "benign",
        "text": "Tubular adenoma is a benign breast tumor in BreakHis. It is not cancer.",
    },
    "Ductal carcinoma": {
        "group": "Malignant",
        "kind": "malignant",
        "text": "Ductal carcinoma is a malignant category in BreakHis.",
    },
    "Lobular carcinoma": {
        "group": "Malignant",
        "kind": "malignant",
        "text": "Lobular carcinoma is a malignant category in BreakHis.",
    },
    "Mucinous carcinoma": {
        "group": "Malignant",
        "kind": "malignant",
        "text": "Mucinous carcinoma is a malignant category in BreakHis.",
    },
    "Papillary carcinoma": {
        "group": "Malignant",
        "kind": "malignant",
        "text": "Papillary carcinoma is a malignant category in BreakHis.",
    },
    "Normal": {
        "group": "Not malignant",
        "kind": "none",
        "text": "Normal is the no-cancer category in this CT dataset. It is not a statement that a person is healthy.",
    },
    "Benign": {
        "group": "Not malignant",
        "kind": "benign",
        "text": "Benign is a finding in this CT dataset. It is not cancer, and it is not the same as a normal slice.",
    },
    "Malignant": {
        "group": "Malignant",
        "kind": "malignant",
        "text": "Malignant is the cancer category in this CT dataset. The label was assigned to the slice by the dataset authors, not by this app.",
    },
}
LABEL_ALIASES = {
    "akiec": "actinic",
    "bcc": "basal",
    "bkl": "benign",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic",
    "nevi": "melanocytic",
    "nevus": "melanocytic",
    "vasc": "vascular",
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "notumor",
    "healthy": "notumor",
}


class TaskBundle:
    def __init__(self, task, classes, train_ds, val_ds, test_ds, class_weights, split_counts, input_size):
        self.task = task
        self.classes = classes
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.class_weights = class_weights
        self.split_counts = split_counts
        self.input_size = input_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def short_label(name: str) -> str:
    return SHORT.get(name, name)


def fold_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def match_class(text: str, classes: list[str]) -> int | None:
    key = fold_text(text or "")
    if not key:
        return None
    for index, name in enumerate(classes):
        options = {fold_text(name), fold_text(short_label(name))}
        if key in options:
            return index
    if key in LABEL_ALIASES:
        needle = LABEL_ALIASES[key]
        hits = [i for i, name in enumerate(classes) if needle in fold_text(name)]
        if len(hits) == 1:
            return hits[0]
    hits = []
    for index, name in enumerate(classes):
        folded = fold_text(name)
        short = fold_text(short_label(name))
        if folded.startswith(key) or short.startswith(key):
            hits.append(index)
    if len(hits) == 1:
        return hits[0]
    return None


def slug(name: str) -> str:
    chars = []
    for char in name.lower():
        if char.isalnum():
            chars.append(char)
        elif char in " -_":
            chars.append("-")
    text = "".join(chars).strip("-")
    while "--" in text:
        text = text.replace("--", "-")
    return text or "class"


def make_transforms(input_size: int):
    train_tf = T.Compose(
        [
            T.Resize((input_size + 32, input_size + 32)),
            T.RandomResizedCrop(input_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    return train_tf, eval_tf


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=tensor.dtype).view(3, 1, 1)
    image = tensor.detach().cpu() * std + mean
    return image.clamp(0, 1)


def to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.size and float(array.max()) <= 1.0:
            array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


class ImageListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


class SkinDataset(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        image = to_pil(image)
        label = int(np.asarray(label).reshape(-1)[0])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class RelabelDataset(Dataset):
    """Same images, with class ids collapsed into a coarser question."""

    def __init__(self, base, mapping: dict[int, int]):
        self.base = base
        self.mapping = {int(key): int(value) for key, value in mapping.items()}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, self.mapping[int(label)]


def unwrap_dataset(dataset):
    while isinstance(dataset, RelabelDataset):
        dataset = dataset.base
    return dataset


def raw_image(dataset, index: int) -> Image.Image:
    dataset = unwrap_dataset(dataset)
    if isinstance(dataset, ImageListDataset):
        with Image.open(dataset.samples[index][0]) as opened:
            return to_pil(opened.convert("RGB"))
    if isinstance(dataset, SkinDataset):
        image, _label = dataset.base[index]
        return to_pil(image)
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def _raw_labels(dataset) -> list[int]:
    dataset = unwrap_dataset(dataset)
    if isinstance(dataset, ImageListDataset):
        return [int(label) for _path, label in dataset.samples]
    if isinstance(dataset, SkinDataset):
        return _skin_targets(dataset.base).tolist()
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def norm_key(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def collect_brain(root: Path):
    found = []
    root = Path(root)
    if not root.exists():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX" and not d.startswith(".")]
        path = Path(dirpath)
        if "__MACOSX" in path.parts:
            continue
        class_name = BRAIN_ALIASES.get(norm_key(path.name))
        if class_name is None:
            continue
        parent = norm_key(path.parent.name)
        if parent in {"training", "train"}:
            hint = "train"
        elif parent in {"testing", "test"}:
            hint = "test"
        elif parent in {"validation", "val"}:
            hint = "val"
        else:
            hint = "all"
        for filename in filenames:
            if filename.startswith("."):
                continue
            if Path(filename).suffix.lower() not in IMAGE_EXTS:
                continue
            found.append((str(path / filename), class_name, hint))
    found.sort(key=lambda item: item[0])
    return found


def peek_tree(root: Path, limit: int = 40) -> str:
    lines = []
    if not root.exists():
        return f"(missing {root})"
    for path in sorted(root.rglob("*"))[:limit]:
        lines.append(str(path.relative_to(root)))
    return "\n".join(lines) if lines else "(empty)"


def is_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    with path.open("rb") as handle:
        return handle.read(2) == b"PK"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "TwoAnswers-coursework/1.0"})
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 20 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if done >= next_report:
                if total:
                    print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
                else:
                    print(f"  {done / 1e6:.0f} MB", flush=True)
                next_report = done + 20 * 1024 * 1024
    temporary.replace(dest)


def ensure_brain_zip() -> Path:
    dest = ROOT / "data" / "brain-tumor-mri-dataset.zip"
    if is_zip(dest):
        return dest
    if dest.exists():
        dest.unlink()
    errors = []
    for url in BRAIN_URLS:
        try:
            download_file(url, dest)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if dest.exists():
                dest.unlink()
            continue
        if is_zip(dest):
            return dest
        errors.append(f"{url}: downloaded file is not a zip")
        if dest.exists():
            dest.unlink()
    raise RuntimeError("Could not download the brain MRI dataset.\n" + "\n".join(errors))


def safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive.infolist():
        target = (dest / member.filename).resolve()
        if not target.is_relative_to(dest):
            raise RuntimeError(f"Unsafe path in zip: {member.filename}")
    archive.extractall(dest)


def brain_image_root() -> Path:
    extracted = ROOT / "data" / "brain"
    if collect_brain(extracted):
        return extracted
    zip_path = ensure_brain_zip()
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        safe_extract(archive, extracted)
    if not collect_brain(extracted):
        raise RuntimeError(
            "The brain dataset zip did not contain class folders.\n" + peek_tree(extracted)
        )
    return extracted


def _split_samples(samples, test_size, seed):
    labels = [label for _path, label in samples]
    indices = np.arange(len(samples))
    try:
        left, right = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=labels
        )
    except ValueError as exc:
        print(f"Stratified split was not possible ({exc}). Using a random split.", flush=True)
        left, right = train_test_split(indices, test_size=test_size, random_state=seed)
    return [samples[i] for i in left], [samples[i] for i in right]


def make_class_weights(labels, n_classes):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    weights = weights / max(weights.mean(), 1e-8)
    return torch.tensor(weights, dtype=torch.float32), counts


def _count_map(samples, classes):
    labels = [label for _path, label in samples]
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(classes))
    return {classes[i]: int(counts[i]) for i in range(len(classes))}


def prepare_brain(input_size: int = INPUT_SIZE) -> TaskBundle:
    found = collect_brain(brain_image_root())
    if not found:
        raise RuntimeError("No brain MRI images were found.")
    present = [name for name in BRAIN_ORDER if any(item[1] == name for item in found)]
    classes = [BRAIN_DISPLAY[name] for name in present]
    name_to_index = {name: index for index, name in enumerate(present)}
    labeled = [(path, name_to_index[name], hint) for path, name, hint in found]
    test_samples = [(path, label) for path, label, hint in labeled if hint == "test"]
    val_samples = [(path, label) for path, label, hint in labeled if hint == "val"]
    train_pool = [(path, label) for path, label, hint in labeled if hint in {"train", "all"}]
    if not test_samples:
        train_pool, test_samples = _split_samples(
            [(path, label) for path, label, _hint in labeled], 0.15, SEED
        )
        train_samples, val_samples = _split_samples(train_pool, 0.15 / 0.85, SEED)
    else:
        if val_samples:
            train_samples = train_pool
        else:
            train_samples, val_samples = _split_samples(train_pool, 0.15, SEED)
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError(
            f"Brain split is empty (train {len(train_samples)}, val {len(val_samples)}, test {len(test_samples)})."
        )
    train_tf, eval_tf = make_transforms(input_size)
    weights, _counts = make_class_weights([label for _path, label in train_samples], len(classes))
    split_counts = {
        "train": _count_map(train_samples, classes),
        "val": _count_map(val_samples, classes),
        "test": _count_map(test_samples, classes),
    }
    return TaskBundle(
        task="brain",
        classes=classes,
        train_ds=ImageListDataset(train_samples, train_tf),
        val_ds=ImageListDataset(val_samples, eval_tf),
        test_ds=ImageListDataset(test_samples, eval_tf),
        class_weights=weights,
        split_counts=split_counts,
        input_size=input_size,
    )


def _medmnist_classes(label_map) -> list[str]:
    classes = []
    for index in range(len(label_map)):
        if index in label_map:
            classes.append(str(label_map[index]))
        else:
            classes.append(str(label_map[str(index)]))
    return classes


def _skin_targets(base) -> np.ndarray:
    return np.asarray(base.labels).reshape(-1).astype(int)


def prepare_skin(input_size: int = INPUT_SIZE) -> TaskBundle:
    from medmnist import INFO, DermaMNIST

    root = ROOT / "data" / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    classes = _medmnist_classes(INFO["dermamnist"]["label"])
    train_tf, eval_tf = make_transforms(input_size)
    bases = {}
    datasets = {}
    for split in ("train", "val", "test"):
        base = DermaMNIST(
            split=split,
            download=True,
            size=input_size,
            root=str(root),
            as_rgb=True,
            mmap_mode="r",
        )
        bases[split] = base
        datasets[split] = SkinDataset(base, train_tf if split == "train" else eval_tf)
    weights, _counts = make_class_weights(_skin_targets(bases["train"]), len(classes))
    split_counts = {}
    for split, base in bases.items():
        counts = np.bincount(_skin_targets(base), minlength=len(classes))
        split_counts[split] = {classes[i]: int(counts[i]) for i in range(len(classes))}
    return TaskBundle(
        task="skin",
        classes=classes,
        train_ds=datasets["train"],
        val_ds=datasets["val"],
        test_ds=datasets["test"],
        class_weights=weights,
        split_counts=split_counts,
        input_size=input_size,
    )


def _count_from_labels(labels, classes) -> dict:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(classes))
    return {classes[i]: int(counts[i]) for i in range(len(classes))}


def relabel_bundle(bundle: TaskBundle, task_name: str, classes: list[str], mapping: dict[int, int]) -> TaskBundle:
    def mapped(dataset):
        return [int(mapping[int(label)]) for label in _raw_labels(dataset)]

    train_labels = mapped(bundle.train_ds)
    weights, _counts = make_class_weights(train_labels, len(classes))
    split_counts = {
        "train": _count_from_labels(train_labels, classes),
        "val": _count_from_labels(mapped(bundle.val_ds), classes),
        "test": _count_from_labels(mapped(bundle.test_ds), classes),
    }
    return TaskBundle(
        task=task_name,
        classes=classes,
        train_ds=RelabelDataset(bundle.train_ds, mapping),
        val_ds=RelabelDataset(bundle.val_ds, mapping),
        test_ds=RelabelDataset(bundle.test_ds, mapping),
        class_weights=weights,
        split_counts=split_counts,
        input_size=bundle.input_size,
    )


def prepare_brain_binary(input_size: int = INPUT_SIZE) -> TaskBundle:
    bundle = prepare_brain(input_size)
    mapping = {index: (1 if name == "No tumor" else 0) for index, name in enumerate(bundle.classes)}
    return relabel_bundle(bundle, "brain_binary", BINARY_CLASSES["brain"], mapping)


def prepare_skin_binary(input_size: int = INPUT_SIZE) -> TaskBundle:
    bundle = prepare_skin(input_size)
    cancer_related = {
        name
        for name, info in CLASS_MEANING.items()
        if info["group"] == "Cancer-related"
    }
    mapping = {
        index: (0 if name in cancer_related else 1)
        for index, name in enumerate(bundle.classes)
    }
    missing = [name for name in bundle.classes if name not in CLASS_MEANING]
    if missing:
        raise RuntimeError(f"Skin labels have no cancer/benign meaning: {missing}")
    return relabel_bundle(bundle, "skin_binary", BINARY_CLASSES["skin"], mapping)


BREAKHIS_URL = "https://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
BREAKHIS_BYTES = 4273561758
LUNG_URL = "https://codeload.github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset/zip/refs/heads/main"
BREAST_CLASS = {
    "adenosis": "Adenosis",
    "fibroadenoma": "Fibroadenoma",
    "phyllodestumor": "Phyllodes tumor",
    "tubularadenoma": "Tubular adenoma",
    "ductalcarcinoma": "Ductal carcinoma",
    "lobularcarcinoma": "Lobular carcinoma",
    "mucinouscarcinoma": "Mucinous carcinoma",
    "papillarycarcinoma": "Papillary carcinoma",
}
BREAST_ORDER = [
    "Adenosis",
    "Fibroadenoma",
    "Phyllodes tumor",
    "Tubular adenoma",
    "Ductal carcinoma",
    "Lobular carcinoma",
    "Mucinous carcinoma",
    "Papillary carcinoma",
]
LUNG_CLASS = {
    "normal": "Normal",
    "normalcases": "Normal",
    "benign": "Benign",
    "benigncases": "Benign",
    "bengin": "Benign",
    "bengincases": "Benign",
    "malignant": "Malignant",
    "malignantcases": "Malignant",
}
LUNG_ORDER = ["Normal", "Benign", "Malignant"]


def _samples_from_groups(grouped, classes, group_ids):
    name_to_index = {name: index for index, name in enumerate(classes)}
    samples = []
    for group_id in group_ids:
        label = name_to_index[grouped[group_id]["class"]]
        for path in grouped[group_id]["paths"]:
            samples.append((path, label))
    samples.sort(key=lambda item: item[0])
    return samples


def split_patient_records(records, classes, test_fraction=0.15, val_fraction=0.15, seed=SEED):
    """Keep every image from one patient in a single split.

    records are (path, class_name, patient_id). A class with fewer than three patients
    stays in training: one patient cannot fill train, validation, and test.
    """
    grouped = {}
    for path, class_name, patient_id in records:
        slot = grouped.setdefault(patient_id, {"class": class_name, "paths": []})
        if slot["class"] != class_name:
            raise RuntimeError(f"Patient {patient_id} has more than one class.")
        slot["paths"].append(path)
    by_class = {name: [] for name in classes}
    for patient_id in sorted(grouped):
        by_class[grouped[patient_id]["class"]].append(patient_id)
    rng = np.random.RandomState(seed)
    train_groups, val_groups, test_groups = [], [], []
    for class_name in classes:
        members = by_class[class_name]
        rng.shuffle(members)
        count = len(members)
        if count == 0:
            continue
        if count < 3:
            train_groups.extend(members)
            print(
                f"  {class_name}: {count} patient(s), kept in training "
                "(a patient-level validation and test split needs at least 3 patients).",
                flush=True,
            )
            continue
        n_test = max(1, int(round(count * test_fraction)))
        n_val = max(1, int(round(count * val_fraction)))
        while n_test + n_val >= count:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        test_groups.extend(members[:n_test])
        val_groups.extend(members[n_test:n_test + n_val])
        train_groups.extend(members[n_test + n_val:])
    return (
        _samples_from_groups(grouped, classes, train_groups),
        _samples_from_groups(grouped, classes, val_groups),
        _samples_from_groups(grouped, classes, test_groups),
        {"train": len(train_groups), "val": len(val_groups), "test": len(test_groups)},
    )


def _image_splits(samples):
    train_val, test_samples = _split_samples(samples, 0.15, SEED)
    train_samples, val_samples = _split_samples(train_val, 0.15 / 0.85, SEED)
    return train_samples, val_samples, test_samples


def _bundle_from_samples(task, classes, train_samples, val_samples, test_samples, input_size):
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError(
            f"{task} split is empty (train {len(train_samples)}, val {len(val_samples)}, test {len(test_samples)})."
        )
    train_tf, eval_tf = make_transforms(input_size)
    weights, _counts = make_class_weights([label for _path, label in train_samples], len(classes))
    return TaskBundle(
        task=task,
        classes=classes,
        train_ds=ImageListDataset(train_samples, train_tf),
        val_ds=ImageListDataset(val_samples, eval_tf),
        test_ds=ImageListDataset(test_samples, eval_tf),
        class_weights=weights,
        split_counts={
            "train": _count_map(train_samples, classes),
            "val": _count_map(val_samples, classes),
            "test": _count_map(test_samples, classes),
        },
        input_size=input_size,
    )


def _binary_from_groups(bundle, task_name, group_name):
    missing = [name for name in bundle.classes if name not in CLASS_MEANING]
    if missing:
        raise RuntimeError(f"{bundle.task} labels have no benign/malignant meaning: {missing}")
    mapping = {
        index: (0 if category_meaning(name)["group"] == group_name else 1)
        for index, name in enumerate(bundle.classes)
    }
    return relabel_bundle(bundle, task_name, BINARY_CLASSES[bundle.task], mapping)


def collect_named_folders(root: Path, class_map: dict[str, str]):
    found = []
    root = Path(root)
    if not root.exists():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != "__MACOSX" and not name.startswith(".")]
        path = Path(dirpath)
        if "__MACOSX" in path.parts:
            continue
        class_name = class_map.get(norm_key(path.name))
        if class_name is None:
            continue
        for filename in filenames:
            if filename.startswith("."):
                continue
            if Path(filename).suffix.lower() not in IMAGE_EXTS:
                continue
            found.append((str(path / filename), class_name, path.name))
    found.sort(key=lambda item: item[0])
    return found


def collect_breakhis(root: Path):
    found = []
    root = Path(root)
    if not root.exists():
        return found
    magnifications = {"40x", "100x", "200x", "400x"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != "__MACOSX" and not name.startswith(".")]
        path = Path(dirpath)
        if "__MACOSX" in path.parts:
            continue
        if norm_key(path.name) not in magnifications:
            continue
        patient_dir = path.parent
        class_dir = patient_dir.parent
        class_name = BREAST_CLASS.get(norm_key(class_dir.name))
        if class_name is None:
            continue
        patient_id = patient_dir.name
        for filename in filenames:
            if filename.startswith(".") or Path(filename).suffix.lower() not in IMAGE_EXTS:
                continue
            found.append((str(path / filename), class_name, patient_id))
    found.sort(key=lambda item: item[0])
    return found


def breakhis_archive() -> Path:
    dest = ROOT / "data" / "BreaKHis_v1.tar.gz"
    if dest.exists() and dest.stat().st_size == BREAKHIS_BYTES:
        return dest
    if dest.exists() and dest.stat().st_size > BREAKHIS_BYTES:
        dest.unlink()
    download_file(BREAKHIS_URL, dest)
    if not dest.exists() or dest.stat().st_size != BREAKHIS_BYTES:
        raise RuntimeError(
            f"BreakHis download is incomplete ({dest.stat().st_size if dest.exists() else 0} bytes, "
            f"expected {BREAKHIS_BYTES})."
        )
    return dest


def breakhis_image_root() -> Path:
    extracted = ROOT / "data" / "breakhis"
    if collect_breakhis(extracted):
        return extracted
    archive = breakhis_archive()
    extracted.mkdir(parents=True, exist_ok=True)
    print("Extracting BreakHis. This is a few gigabytes.", flush=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(extracted, filter="data")
    if not collect_breakhis(extracted):
        raise RuntimeError("The BreakHis archive did not contain patient image folders.\n" + peek_tree(extracted))
    return extracted


def prepare_breast(input_size: int = INPUT_SIZE) -> TaskBundle:
    found = collect_breakhis(breakhis_image_root())
    if not found:
        raise RuntimeError("No BreakHis images were found.")
    classes = [name for name in BREAST_ORDER if any(item[1] == name for item in found)]
    print(f"BreakHis patients are split so one patient stays in one set. Images: {len(found)}.", flush=True)
    train_samples, val_samples, test_samples, patients = split_patient_records(found, classes)
    print(
        f"  patients train {patients['train']}, val {patients['val']}, test {patients['test']}",
        flush=True,
    )
    return _bundle_from_samples("breast", classes, train_samples, val_samples, test_samples, input_size)


def prepare_breast_binary(input_size: int = INPUT_SIZE) -> TaskBundle:
    return _binary_from_groups(prepare_breast(input_size), "breast_binary", "Benign")


def lung_image_root() -> Path:
    extracted = ROOT / "data" / "iqoth"
    if collect_named_folders(extracted, LUNG_CLASS):
        return extracted
    archive = ROOT / "data" / "iqoth-main.zip"
    if not archive.exists() or archive.stat().st_size < 1000:
        download_file(LUNG_URL, archive)
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        safe_extract(handle, extracted)
    if not collect_named_folders(extracted, LUNG_CLASS):
        raise RuntimeError("The lung CT zip did not contain class folders.\n" + peek_tree(extracted))
    return extracted


def prepare_lung(input_size: int = INPUT_SIZE) -> TaskBundle:
    found = collect_named_folders(lung_image_root(), LUNG_CLASS)
    if not found:
        raise RuntimeError("No lung CT images were found.")
    classes = [name for name in LUNG_ORDER if any(item[1] == name for item in found)]
    name_to_index = {name: index for index, name in enumerate(classes)}
    samples = [(path, name_to_index[class_name]) for path, class_name, _folder in found]
    print(
        "Lung CT filenames do not contain a patient id, so the split is by image. "
        "Slices from one scan can still land in more than one set.",
        flush=True,
    )
    train_samples, val_samples, test_samples = _image_splits(samples)
    return _bundle_from_samples("lung", classes, train_samples, val_samples, test_samples, input_size)


def prepare_lung_binary(input_size: int = INPUT_SIZE) -> TaskBundle:
    return _binary_from_groups(prepare_lung(input_size), "lung_binary", "Not malignant")


def prepare_task(task: str, input_size: int = INPUT_SIZE) -> TaskBundle:
    if task == "brain":
        return prepare_brain(input_size)
    if task == "skin":
        return prepare_skin(input_size)
    if task == "brain_binary":
        return prepare_brain_binary(input_size)
    if task == "skin_binary":
        return prepare_skin_binary(input_size)
    if task == "breast":
        return prepare_breast(input_size)
    if task == "breast_binary":
        return prepare_breast_binary(input_size)
    if task == "lung":
        return prepare_lung(input_size)
    if task == "lung_binary":
        return prepare_lung_binary(input_size)
    raise ValueError(f"Unknown task: {task}")


def category_meaning(class_name: str) -> dict:
    info = CLASS_MEANING.get(class_name)
    if info is not None:
        return info
    return {
        "group": class_name,
        "kind": "unknown",
        "text": "This category has no extra benign or malignant note in the dataset.",
    }


def describe_counts(bundle: TaskBundle) -> None:
    print(f"\n{TASK_TITLES[bundle.task]} splits", flush=True)
    for split, counts in bundle.split_counts.items():
        detail = ", ".join(f"{short_label(name)} {n}" for name, n in counts.items())
        print(f"  {split}: {sum(counts.values())} ({detail})", flush=True)
    weights = ", ".join(
        f"{short_label(name)} {weight:.2f}"
        for name, weight in zip(bundle.classes, bundle.class_weights.tolist())
    )
    print(f"  class weights (mean 1): {weights}", flush=True)


def build_resnet(n_classes: int, freeze_backbone: bool = False, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=DROPOUT), nn.Linear(in_features, n_classes))
    return model


def disable_inplace_relu(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def apply_train_mode(model: nn.Module, freeze_backbone: bool) -> None:
    model.train()
    if freeze_backbone:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, batch: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        images = batch.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(images)
            score = logits[0, int(class_idx)]
            score.backward()
        gradients = self.gradients
        activations = self.activations
        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM did not receive activations. The chosen layer may be frozen.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        result = cam.detach().cpu().numpy()
        self.activations = None
        self.gradients = None
        return result


def prepare_for_explanation(model: nn.Module) -> GradCAM:
    model.eval()
    disable_inplace_relu(model)
    return GradCAM(model, model.layer4[-1])


def make_loader(dataset, batch_size, shuffle, seed, device):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )


def _move_batch(x, y, device):
    return x.to(device), y.view(-1).long().to(device)


def train_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    device,
    epochs,
    lr,
    weight_decay,
    patience,
    freeze_backbone,
    task_name,
    config_name,
    max_batches=None,
):
    model.to(device)
    weights = class_weights.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history = {key: [] for key in ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1")}
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_f1 = -1.0
    best_epoch = 0
    best_val = {}
    patience_left = patience
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        apply_train_mode(model, freeze_backbone)
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch_index, (images, labels) in enumerate(tqdm(train_loader, desc=f"{task_name} {config_name} {epoch}/{epochs}", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = _move_batch(images, labels, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
        train_loss = total_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)

        val_loss, y_true, y_pred = run_eval(
            model, val_loader, device, class_weights, max_batches=max_batches, return_probs=False
        )
        class_labels = list(range(int(class_weights.numel())))
        val_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
        val_f1 = (
            float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0))
            if y_true
            else 0.0
        )
        improved = best_epoch == 0 or val_f1 > best_f1 + MIN_DELTA
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val = {
                "loss": float(val_loss),
                "accuracy": val_acc,
                "macro_f1": val_f1,
                "precision_macro": float(
                    precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
                ),
                "recall_macro": float(
                    recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
                ),
            }
            patience_left = patience
        else:
            patience_left -= 1
        history["epoch"].append(epoch)
        history["lr"].append(float(current_lr))
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        marker = "  saved" if improved else ""
        print(
            f"[{task_name} {config_name}] epoch {epoch}/{epochs}  "
            f"lr {current_lr:.2e}  train loss {train_loss:.4f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} macro-F1 {val_f1:.3f}{marker}",
            flush=True,
        )
        scheduler.step()
        if patience_left <= 0:
            print(f"[{task_name} {config_name}] early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.to(device)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "training_seconds": time.perf_counter() - started,
        "epochs_ran": len(history["epoch"]),
    }


def run_eval(model, loader, device, class_weights=None, max_batches=None, return_probs=False):
    model.eval()
    if class_weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0
    seen = 0
    y_true = []
    y_pred = []
    chunks = []
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = _move_batch(images, labels, device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            if return_probs:
                chunks.append(torch.softmax(logits, dim=1).detach().cpu())
    mean_loss = total_loss / max(seen, 1)
    if return_probs:
        probs = torch.cat(chunks, dim=0).numpy() if chunks else np.zeros((0, 0), dtype=np.float32)
        return mean_loss, y_true, y_pred, probs
    return mean_loss, y_true, y_pred


def classification_metrics(y_true, y_pred, probs, classes) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probs = np.asarray(probs)
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            classes[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=classes, digits=4, zero_division=0
        ),
    }
    try:
        if probs.ndim == 2 and probs.shape[1] == 2 and len(np.unique(y_true)) == 2:
            # Binary AUC is symmetric, so one column is enough. Passing both columns
            # makes recent scikit-learn reject the call.
            metrics["roc_auc_macro_ovr"] = float(roc_auc_score(y_true, probs[:, 1]))
        elif probs.ndim == 2 and probs.shape[1] > 2:
            metrics["roc_auc_macro_ovr"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
        else:
            metrics["roc_auc_macro_ovr"] = None
    except ValueError:
        metrics["roc_auc_macro_ovr"] = None
    return metrics


def most_confused(matrix, classes):
    confusion = np.asarray(matrix, dtype=float)
    if confusion.size == 0:
        return None
    off = confusion.copy()
    np.fill_diagonal(off, 0)
    if off.sum() == 0:
        return None
    true_i, pred_j = np.unravel_index(np.argmax(off), off.shape)
    return {
        "true": classes[int(true_i)],
        "predicted": classes[int(pred_j)],
        "count": int(off[true_i, pred_j]),
    }


def benchmark_latency(model, dataset, device, input_size, repeats: int = 30) -> float:
    model.eval()
    image = raw_image(dataset, 0)
    batch = make_transforms(input_size)[1](image).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(batch)
        sync(device)
        times = []
        for _ in range(repeats):
            sync(device)
            started = time.perf_counter()
            model(batch)
            sync(device)
            times.append(time.perf_counter() - started)
    times.sort()
    return float(times[len(times) // 2]) * 1000.0


def benchmark_batched(model, loader, device) -> float:
    model.eval()
    images, _labels = next(iter(loader))
    warmup = images.to(device)
    with torch.no_grad():
        for _ in range(3):
            model(warmup)
        sync(device)
        elapsed = 0.0
        seen = 0
        for images, _labels in loader:
            images = images.to(device)
            sync(device)
            started = time.perf_counter()
            model(images)
            sync(device)
            elapsed += time.perf_counter() - started
            seen += images.size(0)
    return elapsed / max(seen, 1) * 1000.0


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_curves(history, best_epoch, path, title) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation accuracy")
    axes[1].plot(epochs, history["val_f1"], label="Validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy and macro-F1")
    axes[1].legend()
    for axis in axes:
        axis.axvline(best_epoch, color="gray", ls="--", lw=1)
        axis.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, Path(path))


def save_confusion(matrix, classes, path, title) -> None:
    counts = np.asarray(matrix, dtype=float)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    labels = [short_label(name) for name in classes]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for axis, mat, kind in ((axes[0], counts, "Counts"), (axes[1], shares, "Share of true class")):
        image = axis.imshow(mat, cmap="Blues")
        fig.colorbar(image, ax=axis, fraction=0.046)
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=40, ha="right")
        axis.set_yticklabels(labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(kind)
        thresh = mat.max() * 0.62 if mat.max() > 0 else 0
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                text = f"{int(mat[row, col])}" if kind == "Counts" else f"{mat[row, col] * 100:.0f}%"
                color = "white" if mat[row, col] > thresh else "black"
                axis.text(col, row, text, ha="center", va="center", color=color, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, Path(path))


def save_balance(split_counts, path, title) -> None:
    train_counts = split_counts["train"]
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.bar([short_label(name) for name in train_counts], list(train_counts.values()), color="#1f4e79")
    ax.set_ylabel("Training images")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, Path(path))


def save_comparison(rows, path, title) -> None:
    labels = [row["config"] for row in rows]
    series = {
        "Validation macro-F1": [row["best_val"]["macro_f1"] for row in rows],
        "Test macro-F1": [row["test"]["f1_macro"] for row in rows],
        "Test accuracy": [row["test"]["accuracy"] for row in rows],
    }
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for index, (name, values) in enumerate(series.items()):
        ax.bar(x + (index - 1) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, Path(path))


def save_per_class(metrics, classes, path, title) -> None:
    labels = [short_label(name) for name in classes]
    x = np.arange(len(labels))
    width = 0.38
    recall = [metrics["per_class"][name]["recall"] for name in classes]
    f1 = [metrics["per_class"][name]["f1"] for name in classes]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x - width / 2, recall, width, label="Recall")
    ax.bar(x + width / 2, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, Path(path))


def pick_five(y_true, y_pred, probs) -> list[int]:
    correct = [i for i in range(len(y_true)) if int(y_true[i]) == int(y_pred[i])]
    wrong = [i for i in range(len(y_true)) if int(y_true[i]) != int(y_pred[i])]
    correct.sort(key=lambda i: -float(probs[i, int(y_pred[i])]))
    wrong.sort(key=lambda i: -float(probs[i, int(y_pred[i])]))
    chosen = []
    seen = set()
    for index in correct:
        label = int(y_true[index])
        if label in seen:
            continue
        chosen.append(index)
        seen.add(label)
        if len(chosen) == 3:
            break
    for index in wrong:
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == 5:
            break
    for index in correct:
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == 5:
            break
    return chosen[:5]


def save_case_grid(dataset, classes, indices, y_true, y_pred, probs, path, title) -> None:
    n = len(indices)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.5))
    if n == 1:
        axes = [axes]
    for axis, index in zip(axes, indices):
        image, label = dataset[index]
        shown = denormalize(image).permute(1, 2, 0).numpy()
        axis.imshow(shown)
        pred = int(y_pred[index])
        ok = pred == int(label)
        color = "#1b7f3a" if ok else "#b42318"
        axis.set_title(
            f"True: {short_label(classes[int(label)])}\n"
            f"Pred: {short_label(classes[pred])}\n"
            f"{probs[index, pred] * 100:.0f}%",
            color=color,
            fontsize=8,
        )
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, Path(path))


def overlay_cam(image_01, cam) -> Image.Image:
    image = np.clip(np.asarray(image_01), 0, 1)
    heat_values = np.asarray(cam, dtype=np.float32)
    if heat_values.shape != image.shape[:2]:
        resized = Image.fromarray((np.clip(heat_values, 0, 1) * 255).astype(np.uint8))
        resized = resized.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
        heat_values = np.asarray(resized).astype(np.float32) / 255.0
    heat = cm.jet(heat_values)[..., :3]
    mixed = np.clip(0.58 * image + 0.42 * heat, 0, 1)
    return Image.fromarray((mixed * 255).astype(np.uint8))


def save_gradcam_grid(model, dataset, classes, indices, y_true, device, input_size, path, title) -> None:
    n = len(indices)
    if n == 0:
        return
    engine = prepare_for_explanation(model)
    transform = make_transforms(input_size)[1]
    fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 5.6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    try:
        for col, index in enumerate(indices):
            batch = transform(raw_image(dataset, index)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(batch)
                pred = int(logits.argmax(dim=1).item())
                confidence = float(torch.softmax(logits, dim=1)[0, pred].item())
            heat = engine(batch, pred)
            shown = denormalize(batch[0]).permute(1, 2, 0).numpy()
            true_index = int(y_true[index])
            ok = pred == true_index
            color = "#1b7f3a" if ok else "#b42318"
            axes[0, col].imshow(shown)
            axes[0, col].set_title(f"True: {short_label(classes[true_index])}", fontsize=8)
            axes[1, col].imshow(np.asarray(overlay_cam(shown, heat)))
            axes[1, col].set_title(
                f"Pred: {short_label(classes[pred])}\n{confidence * 100:.0f}%",
                color=color,
                fontsize=8,
            )
            for axis in (axes[0, col], axes[1, col]):
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        _save(fig, Path(path))
    finally:
        engine.close()
        plt.close(fig)


def render_scores(classes, probs) -> Image.Image:
    order = np.argsort(probs)
    labels = [short_label(classes[i]) for i in order]
    values = np.asarray(probs)[order] * 100
    colors = ["#1f4e79"] * len(order)
    if colors:
        colors[-1] = "#c47b2b"
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(classes) + 1.15))
    try:
        ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score (%)")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return _figure_to_pil(fig)
    finally:
        plt.close(fig)


def _figure_to_pil(fig) -> Image.Image:
    import io

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    return image


def predict_pil(model, image, classes, device, gradcam, input_size) -> dict:
    image = to_pil(image)
    batch = make_transforms(input_size)[1](image).unsqueeze(0).to(device)
    model.eval()
    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(batch)
    sync(device)
    elapsed = time.perf_counter() - started
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    overlay = None
    if gradcam is not None:
        heat = gradcam(batch, pred)
        shown = denormalize(batch[0]).permute(1, 2, 0).numpy()
        overlay = overlay_cam(shown, heat)
    return {
        "pred": pred,
        "probs": probs,
        "seconds": elapsed,
        "overlay": overlay,
        "scores": render_scores(classes, probs),
    }


def load_for_inference(path, device=None):
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_resnet(len(checkpoint["classes"]), freeze_backbone=False, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    disable_inplace_relu(model)
    gradcam = GradCAM(model, model.layer4[-1])
    return model, checkpoint, gradcam, device


def export_class_examples(dataset, classes, y_true, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for index, label in enumerate(y_true):
        label = int(label)
        if label not in chosen:
            chosen[label] = index
    paths = []
    for label, index in sorted(chosen.items()):
        path = dest / f"{label}_{slug(classes[label])}.jpg"
        raw_image(dataset, index).convert("RGB").save(path, quality=90)
        paths.append(path)
    return paths


def heldout_records(dataset, classes, indices, y_true, y_pred, probs) -> list[dict]:
    rows = []
    for index in indices:
        pred = int(y_pred[index])
        true_index = int(y_true[index])
        row = {
            "index": int(index),
            "true": classes[true_index],
            "predicted": classes[pred],
            "confidence": float(probs[index, pred]),
            "correct": pred == true_index,
        }
        source = unwrap_dataset(dataset)
        if isinstance(source, ImageListDataset):
            row["file"] = source.samples[index][0]
        rows.append(row)
    return rows
