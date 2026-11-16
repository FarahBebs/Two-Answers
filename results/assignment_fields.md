# Assignment fields

Numbers below are written by train.py from the finished runs.
Model choice uses validation macro-F1. Test numbers are from the restored best epoch.

| Field | Response |
|---|---|
| Project Title | Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images |
| Repository | https://github.com/FarahBebs/Deep-Learning-Homework-2 |
| Problem / Task | Image classification, with one model per image type |
| Deployment Technology | Gradio |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |

## Brain MRI

| Field | Response |
|---|---|
| Dataset | Brain Tumor MRI Dataset (Nickparvar; images from figshare, SARTAJ, and Br35H) |
| Dataset URL | https://doi.org/10.5281/zenodo.12735702 (also https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Model Architecture | ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head |
| Input Size | 224 × 224 RGB |
| Hyperparameters | batch 32, learning rate 0.0001, weight decay 0.0001, dropout 0.3, up to 8 epochs, early stopping patience 3 on validation macro-F1 (minimum improvement 0.001), seed 42. Selected configuration: finetune, restored epoch 6 of 8 run. Augmentation: resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter. |
| Optimizer | AdamW, cosine decay from the initial learning rate to 1e-6 |
| Activation / Output Function | ReLU inside the network. linear logits; softmax is applied only to report scores. |
| Loss Function | class-weighted cross-entropy (weights average to 1) |
| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). IoU and mAP are not used because this module is classification, not detection or segmentation. |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |
| Best Performance | Test accuracy 99.08%, macro precision 99.07%, macro recall 99.00%, macro F1 99.03%, weighted F1 99.09%, macro ROC-AUC 99.97%. Validation macro-F1 at the chosen epoch: 98.69%. |
| Inference Time | 3.8 ms per image (single image); 1.2 ms per image in batches of 32 |
| Training Time | 5 min 59 s (359.5 s) |
| Deployment Technology | Gradio |

## Brain MRI — tumor or no tumor

| Field | Response |
|---|---|
| Dataset | Same Brain Tumor MRI images, with labels collapsed to two answers: Tumor detected (glioma, meningioma, or pituitary tumor) and No tumor detected. The source labels do not say whether a tumor is benign or malignant. |
| Dataset URL | https://doi.org/10.5281/zenodo.12735702 (also https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Model Architecture | ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head |
| Input Size | 224 × 224 RGB |
| Hyperparameters | batch 32, learning rate 0.0001, weight decay 0.0001, dropout 0.3, up to 8 epochs, early stopping patience 3 on validation macro-F1 (minimum improvement 0.001), seed 42. Selected configuration: finetune, restored epoch 5 of 8 run. Augmentation: resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter. |
| Optimizer | AdamW, cosine decay from the initial learning rate to 1e-6 |
| Activation / Output Function | ReLU inside the network. linear logits; softmax is applied only to report scores. |
| Loss Function | class-weighted cross-entropy (weights average to 1) |
| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). IoU and mAP are not used because this module is classification, not detection or segmentation. |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |
| Best Performance | Test accuracy 99.69%, macro precision 99.64%, macro recall 99.64%, macro F1 99.64%, weighted F1 99.69%, macro ROC-AUC 99.95%. Validation macro-F1 at the chosen epoch: 99.71%. |
| Inference Time | 3.1 ms per image (single image); 1.2 ms per image in batches of 32 |
| Training Time | 5 min 22 s (322.3 s) |
| Deployment Technology | Gradio |

## Breast histopathology

| Field | Response |
|---|---|
| Dataset | BreakHis breast tumor microscopy (Spanhol et al.). Eight subtypes: adenosis, fibroadenoma, phyllodes tumor, tubular adenoma, ductal carcinoma, lobular carcinoma, mucinous carcinoma, and papillary carcinoma. Patients are not split across train, validation, and test. |
| Dataset URL | https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/ (also http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz) |
| Model Architecture | ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head |
| Input Size | 224 × 224 RGB |
| Hyperparameters | batch 32, learning rate 0.0001, weight decay 0.0001, dropout 0.3, up to 8 epochs, early stopping patience 3 on validation macro-F1 (minimum improvement 0.001), seed 42. Selected configuration: finetune, restored epoch 5 of 8 run. Augmentation: resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter. |
| Optimizer | AdamW, cosine decay from the initial learning rate to 1e-6 |
| Activation / Output Function | ReLU inside the network. linear logits; softmax is applied only to report scores. |
| Loss Function | class-weighted cross-entropy (weights average to 1) |
| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). IoU and mAP are not used because this module is classification, not detection or segmentation. |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |
| Best Performance | Test accuracy 60.31%, macro precision 50.33%, macro recall 62.21%, macro F1 54.33%, weighted F1 57.26%, macro ROC-AUC 85.31%. Validation macro-F1 at the chosen epoch: 27.40%. |
| Inference Time | 2.9 ms per image (single image); 1.4 ms per image in batches of 32 |
| Training Time | 10 min 49 s (648.7 s) |
| Deployment Technology | Gradio |

## Breast histopathology — benign or malignant

| Field | Response |
|---|---|
| Dataset | Same BreakHis images, with the eight subtypes collapsed using the dataset's own folders: Benign (adenosis, fibroadenoma, phyllodes tumor, tubular adenoma) and Malignant (ductal, lobular, mucinous, and papillary carcinoma). Phyllodes tumor is benign in this dataset's folder; phyllodes tumors can be malignant outside this file. |
| Dataset URL | https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/ (also http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz) |
| Model Architecture | ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head |
| Input Size | 224 × 224 RGB |
| Hyperparameters | batch 32, learning rate 0.0001, weight decay 0.0001, dropout 0.3, up to 8 epochs, early stopping patience 3 on validation macro-F1 (minimum improvement 0.001), seed 42. Selected configuration: finetune, restored epoch 6 of 8 run. Augmentation: resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter. |
| Optimizer | AdamW, cosine decay from the initial learning rate to 1e-6 |
| Activation / Output Function | ReLU inside the network. linear logits; softmax is applied only to report scores. |
| Loss Function | class-weighted cross-entropy (weights average to 1) |
| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). IoU and mAP are not used because this module is classification, not detection or segmentation. |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |
| Best Performance | Test accuracy 91.80%, macro precision 91.87%, macro recall 91.49%, macro F1 91.65%, weighted F1 91.78%, macro ROC-AUC 96.99%. Validation macro-F1 at the chosen epoch: 91.68%. |
| Inference Time | 2.8 ms per image (single image); 1.3 ms per image in batches of 32 |
| Training Time | 10 min 50 s (650.5 s) |
| Deployment Technology | Gradio |

## Skin lesion (dermoscopy)

| Field | Response |
|---|---|
| Dataset | DermaMNIST (MedMNIST v2), derived from HAM10000 |
| Dataset URL | https://doi.org/10.7910/DVN/DBW86T (also https://medmnist.com/) |
| Model Architecture | ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head |
| Input Size | 224 × 224 RGB |
| Hyperparameters | batch 32, learning rate 0.0001, weight decay 0.0001, dropout 0.3, up to 8 epochs, early stopping patience 3 on validation macro-F1 (minimum improvement 0.001), seed 42. Selected configuration: finetune, restored epoch 6 of 8 run. Augmentation: resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter. |
| Optimizer | AdamW, cosine decay from the initial learning rate to 1e-6 |
| Activation / Output Function | ReLU inside the network. linear logits; softmax is applied only to report scores. |
| Loss Function | class-weighted cross-entropy (weights average to 1) |
| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). IoU and mAP are not used because this module is classification, not detection or segmentation. |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |
| Best Performance | Test accuracy 79.40%, macro precision 67.12%, macro recall 75.70%, macro F1 70.78%, weighted F1 80.52%, macro ROC-AUC 96.38%. Validation macro-F1 at the chosen epoch: 71.66%. |
| Inference Time | 4.1 ms per image (single image); 1.2 ms per image in batches of 32 |
| Training Time | 7 min 29 s (448.6 s) |
| Deployment Technology | Gradio |

## Skin lesion — cancer-related or not

| Field | Response |
|---|---|
| Dataset | Same DermaMNIST images, with labels collapsed to Cancer-related (melanoma, basal cell carcinoma, and actinic keratoses / intraepithelial carcinoma) or Not cancer-related (nevi, benign keratosis, dermatofibroma, vascular lesions). Actinic keratoses are precancerous, not invasive cancer; they are included because the dataset treats them as a carcinoma-spectrum category. |
| Dataset URL | https://doi.org/10.7910/DVN/DBW86T (also https://medmnist.com/) |
| Model Architecture | ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head |
| Input Size | 224 × 224 RGB |
| Hyperparameters | batch 32, learning rate 0.0001, weight decay 0.0001, dropout 0.3, up to 8 epochs, early stopping patience 3 on validation macro-F1 (minimum improvement 0.001), seed 42. Selected configuration: finetune, restored epoch 6 of 8 run. Augmentation: resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter. |
| Optimizer | AdamW, cosine decay from the initial learning rate to 1e-6 |
| Activation / Output Function | ReLU inside the network. linear logits; softmax is applied only to report scores. |
| Loss Function | class-weighted cross-entropy (weights average to 1) |
| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). IoU and mAP are not used because this module is classification, not detection or segmentation. |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |
| Best Performance | Test accuracy 89.78%, macro precision 82.94%, macro recall 87.08%, macro F1 84.74%, weighted F1 90.08%, macro ROC-AUC 94.65%. Validation macro-F1 at the chosen epoch: 83.25%. |
| Inference Time | 3.0 ms per image (single image); 1.2 ms per image in batches of 32 |
| Training Time | 6 min 34 s (393.8 s) |
| Deployment Technology | Gradio |

## Lung CT

| Field | Response |
|---|---|
| Dataset | IQ-OTH/NCCD lung CT slices (alyasriy and AL-Huseiny). Three labels: Normal, Benign, and Malignant. This is slice classification, not 3D nodule detection. LIDC-IDRI was not used: it is about 125 GB of DICOM volumes. A benign slice is not labeled as cancer. |
| Dataset URL | https://doi.org/10.17632/bhmdr45bh2.4 (also https://github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset) |
| Model Architecture | ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head |
| Input Size | 224 × 224 RGB |
| Hyperparameters | batch 32, learning rate 0.0001, weight decay 0.0001, dropout 0.3, up to 8 epochs, early stopping patience 3 on validation macro-F1 (minimum improvement 0.001), seed 42. Selected configuration: finetune, restored epoch 6 of 8 run. Augmentation: resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter. |
| Optimizer | AdamW, cosine decay from the initial learning rate to 1e-6 |
| Activation / Output Function | ReLU inside the network. linear logits; softmax is applied only to report scores. |
| Loss Function | class-weighted cross-entropy (weights average to 1) |
| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). IoU and mAP are not used because this module is classification, not detection or segmentation. |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |
| Best Performance | Test accuracy 96.36%, macro precision 93.17%, macro recall 94.44%, macro F1 93.78%, weighted F1 96.41%, macro ROC-AUC 99.46%. Validation macro-F1 at the chosen epoch: 89.06%. |
| Inference Time | 2.8 ms per image (single image); 1.2 ms per image in batches of 32 |
| Training Time | 1 min 2 s (62.5 s) |
| Deployment Technology | Gradio |

## Lung CT — malignant or not

| Field | Response |
|---|---|
| Dataset | Same IQ-OTH/NCCD slices, collapsed to Not malignant (normal and benign) and Malignant. A benign finding stays on the not-malignant side. |
| Dataset URL | https://doi.org/10.17632/bhmdr45bh2.4 (also https://github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset) |
| Model Architecture | ResNet18 pretrained on ImageNet; all layers fine-tuned; new dropout + linear head |
| Input Size | 224 × 224 RGB |
| Hyperparameters | batch 32, learning rate 0.0001, weight decay 0.0001, dropout 0.3, up to 8 epochs, early stopping patience 3 on validation macro-F1 (minimum improvement 0.001), seed 42. Selected configuration: finetune, restored epoch 3 of 6 run. Augmentation: resize, random resized crop (scale 0.85-1), horizontal flip, rotation ±15°, mild color jitter. |
| Optimizer | AdamW, cosine decay from the initial learning rate to 1e-6 |
| Activation / Output Function | ReLU inside the network. linear logits; softmax is applied only to report scores. |
| Loss Function | class-weighted cross-entropy (weights average to 1) |
| Evaluation Metrics | Accuracy, macro and weighted precision / recall / F1, per-class precision / recall / F1, confusion matrix, macro ROC-AUC (one-vs-rest). IoU and mAP are not used because this module is classification, not detection or segmentation. |
| Training Environment | Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, device mps, Mac14,10, 16 GB RAM |
| Best Performance | Test accuracy 99.39%, macro precision 99.41%, macro recall 99.38%, macro F1 99.39%, weighted F1 99.39%, macro ROC-AUC 100.00%. Validation macro-F1 at the chosen epoch: 99.39%. |
| Inference Time | 2.9 ms per image (single image); 1.2 ms per image in batches of 32 |
| Training Time | 46.6 s |
| Deployment Technology | Gradio |
