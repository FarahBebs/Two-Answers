"""Build the course PDF from the measured results. Not part of training."""

from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Two_Answers_Report.pdf"
RES = ROOT / "results"

pdfmetrics.registerFont(TTFont("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))

NAVY = colors.HexColor("#1B3A4B")
TEAL = colors.HexColor("#1F6F6A")
RULE = colors.HexColor("#D5DEE3")
HEAD_BG = colors.HexColor("#1B3A4B")
ROW = colors.HexColor("#F4F7F8")
NOTE_BG = colors.HexColor("#F3F6F4")


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        "CoverKicker", fontName="Arial", fontSize=9, textColor=TEAL,
        tracking=0.6, spaceAfter=4, leading=12,
    ))
    base.add(ParagraphStyle(
        "CoverTitle", fontName="Arial-Bold", fontSize=18, textColor=NAVY,
        leading=22, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        "CoverSub", fontName="Arial", fontSize=11, textColor=colors.HexColor("#334155"),
        leading=15, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        "H1", fontName="Arial-Bold", fontSize=13, textColor=NAVY,
        spaceBefore=12, spaceAfter=6, leading=16,
    ))
    base.add(ParagraphStyle(
        "H2", fontName="Arial-Bold", fontSize=11, textColor=TEAL,
        spaceBefore=9, spaceAfter=4, leading=14,
    ))
    base.add(ParagraphStyle(
        "Body", fontName="Arial", fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1E293B"), spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        "BulletBody", fontName="Arial", fontSize=9.5, leading=12.5,
        textColor=colors.HexColor("#1E293B"),
    ))
    base.add(ParagraphStyle(
        "Caption", fontName="Arial", fontSize=8, leading=10.5,
        textColor=colors.HexColor("#475569"), spaceBefore=2, spaceAfter=8, alignment=TA_LEFT,
    ))
    base.add(ParagraphStyle(
        "Cell", fontName="Arial", fontSize=8, leading=10.5,
        textColor=colors.HexColor("#1E293B"),
    ))
    base.add(ParagraphStyle(
        "CellHead", fontName="Arial-Bold", fontSize=8, leading=10.5, textColor=colors.white,
    ))
    base.add(ParagraphStyle(
        "Small", fontName="Arial", fontSize=8, leading=10.5,
        textColor=colors.HexColor("#334155"), spaceAfter=4,
    ))
    return base


S = styles()


def P(text, style="Body"):
    return Paragraph(text, S[style])


def bullets(items):
    flow = []
    for item in items:
        flow.append(ListItem(Paragraph(item, S["BulletBody"]), leftIndent=8))
    return ListFlowable(flow, bulletType="bullet", start="•", leftIndent=14, bulletFontName="Arial", bulletFontSize=8, spaceBefore=2, spaceAfter=6)


def field_table(rows):
    data = [[Paragraph("Field", S["CellHead"]), Paragraph("Response", S["CellHead"])]]
    for field, response in rows:
        data.append([Paragraph(field, S["Cell"]), Paragraph(response, S["Cell"])])
    table = Table(data, colWidths=[42 * mm, 128 * mm], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#E7EEF1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (1, i), (1, i), ROW))
    table.setStyle(TableStyle(style_cmds))
    return table


def grid(headers, rows, col_widths):
    data = [[Paragraph(h, S["CellHead"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), S["Cell"]) for c in row])
    table = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), ROW))
    table.setStyle(TableStyle(cmds))
    return table


def mri_figure_for_report() -> Path:
    """Cover the old chart title so it does not contradict the teacher note."""
    src = RES / "brain_five_heldout.png"
    dest = ROOT / "report_screenshots" / "_brain_mri_photos.png"
    dest.parent.mkdir(exist_ok=True)
    im = PILImage.open(src).convert("RGB")
    drawing = ImageDraw.Draw(im)
    # Remove only the old centered title. The True and Pred lines start just below it.
    drawing.rectangle((600, 12, 1280, 31), fill="white")
    im.save(dest)
    return dest


def figure(name, caption, max_w=168 * mm, path=None):
    path = Path(path) if path else RES / name
    with PILImage.open(path) as im:
        w, h = im.size
    height = max_w * h / w
    if height > 95 * mm:
        height = 95 * mm
        max_w = height * w / h
    return KeepTogether([
        Image(str(path), width=max_w, height=height),
        P(caption, "Caption"),
    ])


def note_box(text):
    inner = Paragraph(text, S["Body"])
    table = Table([[inner]], colWidths=[170 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NOTE_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


SHARED = [
    ("Model Architecture", "ResNet18, ImageNet weights, all layers fine-tuned. New head: dropout 0.3, then a linear layer with one logit per class."),
    ("Input Size", "224 x 224 RGB."),
    ("Hyperparameters", "Batch 32, learning rate 1e-4, weight decay 1e-4, dropout 0.3, up to 8 epochs, early stopping patience 3, minimum improvement 0.001 on validation macro-F1, seed 42. Train-only augmentation: resize to 256, random resized crop (scale 0.85 to 1), horizontal flip, rotation +/- 15 degrees, mild color jitter. Evaluation resizes the whole image to 224 x 224."),
    ("Optimizer", "AdamW, cosine decay from 1e-4 down to 1e-6."),
    ("Activation / Output Function", "ReLU inside the ResNet. The head outputs linear logits. Softmax is used only to print scores. Cross-entropy applies log-softmax itself."),
    ("Loss Function", "Class-weighted cross-entropy. Each weight is N / (K x n_c), then the weights are rescaled so they average to 1."),
    ("Evaluation Metrics", "Accuracy, precision, recall, and F1 (macro and weighted), plus a confusion matrix. Macro ROC-AUC is extra. IoU, mIoU, Dice, and mAP are not used: this homework is classification, not detection or segmentation."),
    ("Training Environment", "Python 3.13.2, PyTorch 2.15.0, torchvision 0.29.2, Apple Silicon (Mac14,10), 16 GB RAM, PyTorch MPS."),
]
DEPLOY = ("Deployment Technology", "Gradio. The page is named Two Answers and runs locally at http://127.0.0.1:7860.")


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Arial", 8)
    canvas.drawString(18 * mm, A4[1] - 7.5 * mm, "Computer Vision homework  ·  Deep learning application")
    canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 7.5 * mm, "Farah Alhasan  ·  21120205709")
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont("Arial", 8)
    canvas.drawString(18 * mm, 8 * mm, "Two Answers  ·  not a medical device")
    canvas.drawRightString(A4[0] - 18 * mm, 8 * mm, str(doc.page))
    canvas.restoreState()


def build():
    story = []
    story.append(Spacer(1, 6 * mm))
    story.append(P("COMPUTER VISION – ÖDEVİ", "CoverKicker"))
    story.append(P("Derin Öğrenme Uygulaması", "CoverSub"))
    story.append(P("Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images", "CoverTitle"))
    story.append(P("Student: Farah Alhasan<br/>Student number: 21120205709<br/>Repository: https://github.com/FarahBebs/Two-Answers", "CoverSub"))
    story.append(note_box(
        "<b>This is not a medical device.</b> A score says how strongly a picture matches a training folder. "
        "It is not a diagnosis, not a risk estimate, and not something to use for a health decision. "
        "A tumor is not the same thing as cancer. The brain labels do not say benign or malignant, so that model is not allowed to answer “cancer.”"
    ))
    story.append(Spacer(1, 3 * mm))

    story.append(P("1. Purpose", "H1"))
    story.append(P(
        "The assignment asks for an open dataset, a computer-vision model, a measured result, and a working app. "
        "I trained eight ResNet18 classifiers: a specific category model and a coarser group model for each of four image types. "
        "The category model names the dataset class. The group model answers a shorter question that the labels actually support. "
        "Brain: tumor or no tumor, not cancer or no cancer. Breast: benign or malignant, using BreakHis’s own folders. "
        "Skin: cancer-related or not, and cancer-related includes precancerous actinic keratoses. Lung: malignant or not, and a benign slice stays on the not-malignant side."
    ))

    story.append(P("2. Task", "H1"))
    story.append(P(
        "The task is <b>image classification</b>. I did not do object detection, semantic segmentation, instance segmentation, or face detection. "
        "Each uploaded image is scored by two models for that image type."
    ))
    story.append(grid(
        ["Image type", "Specific category", "Coarser question"],
        [
            ["Brain MRI", "Glioma, meningioma, pituitary tumor, or no tumor", "Tumor detected, or no tumor detected"],
            ["Breast microscopy", "Eight BreakHis tumor subtypes", "Benign or malignant"],
            ["Skin lesion", "Seven HAM10000 categories", "Cancer-related or not"],
            ["Lung CT slice", "Normal, benign, or malignant", "Malignant or not"],
        ],
        [32 * mm, 70 * mm, 68 * mm],
    ))
    story.append(Spacer(1, 2 * mm))

    story.append(P("3. Model architecture", "H1"))
    story.append(P(
        "The architecture is ResNet18, which is one of the architectures named in the assignment. "
        "It is deep enough to start from ImageNet features and small enough to fine-tune on a 16 GB laptop (11.2 million weights). "
        "I also trained a frozen-backbone version on brain and skin, changing only the new head, and kept the run with the higher validation macro-F1. "
        "Fine-tuning won both comparisons, so breast and lung use that same fine-tune recipe only. "
        "The frozen brain head has 2,052 trainable weights. The frozen skin head has 3,591."
    ))

    story.append(P("4. Datasets and splits", "H1"))
    story.append(P(
        "All four datasets are public. Train, validation, and test are separate. The test set is scored only after the epoch is chosen on validation macro-F1. "
        "The example lists in the assignment (MNIST, CIFAR, COCO, and the others) are examples. These medical sets match an image-classification task."
    ))
    story.append(P("Brain MRI", "H2"))
    story.append(P(
        "Brain Tumor MRI Dataset (Nickparvar; figshare, SARTAJ, Br35H), CC BY 4.0. "
        "URL: https://doi.org/10.5281/zenodo.12735702. The official Testing folder is the untouched test set. "
        "The Training folder is split 85/15, stratified, seed 42. There are no patient IDs, so slices from one scan can sit in more than one split, and a score near 99% can be optimistic for a new patient."
    ))
    story.append(grid(
        ["Split", "Images", "Glioma", "Meningioma", "Pituitary", "No tumor"],
        [
            ["Train", "4,855", "1,123", "1,138", "1,238", "1,356"],
            ["Validation", "857", "198", "201", "219", "239"],
            ["Test", "1,311", "300", "306", "300", "405"],
        ],
        [28 * mm, 24 * mm, 28 * mm, 32 * mm, 28 * mm, 30 * mm],
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(P("Skin lesions", "H2"))
    story.append(P(
        "DermaMNIST, size 224, from HAM10000 (Tschandl et al.), CC BY-NC 4.0. "
        "URL: https://doi.org/10.7910/DVN/DBW86T and https://medmnist.com/. I kept the official MedMNIST train, validation, and test split. I did not reshuffle it. "
        "Nevi are 1,341 of 2,005 test images (66.9%). A model that always says nevus would already have 66.9% accuracy. That is why model choice uses macro-F1, not accuracy. "
        "HAM10000 can contain more than one image of the same lesion, so I do not claim a lesion-level split."
    ))
    story.append(grid(
        ["Split", "Images", "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        [
            ["Train", "7,007", "228", "359", "769", "80", "779", "4,693", "99"],
            ["Validation", "1,003", "33", "52", "110", "12", "111", "671", "14"],
            ["Test", "2,005", "66", "103", "220", "23", "223", "1,341", "29"],
        ],
        [24 * mm, 18 * mm, 16 * mm, 16 * mm, 16 * mm, 14 * mm, 16 * mm, 18 * mm, 16 * mm],
    ))
    story.append(P("akiec = actinic keratoses and intraepithelial carcinoma, bcc = basal cell carcinoma, bkl = benign keratosis-like lesions, df = dermatofibroma, mel = melanoma, nv = melanocytic nevi, vasc = vascular lesions.", "Caption"))

    story.append(P("Breast microscopy", "H2"))
    story.append(P(
        "BreakHis v1, 7,909 images, 82 patients, magnifications 40x, 100x, 200x, and 400x. "
        "URL: http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz. "
        "The split is by patient, seed 42: 54 train (4,973 images), 14 validation (1,399), 14 test (1,537). Every image of one patient stays in one split. "
        "Adenosis, fibroadenoma, phyllodes tumor, and tubular adenoma are the benign folder. Ductal, lobular, mucinous, and papillary carcinoma are the malignant folder. "
        "Phyllodes is filed under benign in this dataset. Phyllodes tumors can be malignant outside this file. Training saw only 60 phyllodes images, from one small patient group, and the test side has 235."
    ))
    story.append(P("Lung CT slices", "H2"))
    story.append(P(
        "IQ-OTH/NCCD, 1,097 JPEG slices from 110 cases: 416 normal, 120 benign (the folder is spelled “Bengin cases”), 561 malignant. "
        "URL: https://doi.org/10.17632/bhmdr45bh2.4. License: CC0. "
        "This is slice classification, not a box around a nodule, and it is not LIDC-IDRI. LIDC-IDRI is about 125 GB of 3D CT and is not in this project. "
        "The file names are image indexes, not verified patient IDs, so the split is image-level 70/15/15, seed 42: 767 train, 165 validation, 165 test. Slices from one scan can leak across splits. A score near 99% is not a patient-level result."
    ))

    story.append(P("5. Settings, and why", "H1"))
    story.append(P(
        "Learning rate 1e-4 is small because the network already has ImageNet weights. A larger rate is used only for the frozen head (1e-3), which has to learn the classes from a fixed backbone. "
        "Batch size 32 fits the 16 GB machine. The epoch cap is 8, with patience 3, so a run stops when validation macro-F1 stops improving by at least 0.001, and the best epoch is restored before the test score. "
        "AdamW is used instead of plain Adam because the weight decay (1e-4) is separate from the adaptive step. Cosine decay lowers the rate to 1e-6 so the last epochs take smaller steps. "
        "ReLU is the activation already inside ResNet. The output is logits, not a softmax layer, because cross-entropy expects logits. "
        "Class weights stop the skin loss from being solved by calling every image a nevus. The same weighting is used on the other sets, where the classes are less skewed and the weights stay closer to 1. "
        "Augmentation is train-only. Evaluation does not crop, because a crop would throw away the border of an MRI or a CT slice."
    ))
    story.append(grid(
        ["Configuration", "What is trained", "Learning rate", "Epoch cap", "Used for"],
        [
            ["Fine-tune", "All 11.2 million weights", "1e-4", "8", "All eight selected models"],
            ["Frozen backbone", "New head only", "1e-3", "6", "Brain and skin comparison only"],
        ],
        [32 * mm, 42 * mm, 28 * mm, 22 * mm, 46 * mm],
    ))
    story.append(Spacer(1, 2 * mm))

    story.append(P("6. Performance", "H1"))
    story.append(P(
        "The required classification metrics are accuracy, precision, recall, and F1. I report the macro averages, so a rare class counts as much as a common one, and I also report weighted F1. "
        "Training time is the wall clock of the selected run. Inference time is the median forward pass on one image, and the per-image time inside a batch of 32. Grad-CAM is not included in that timing. "
        "The Gradio page is slower (about 0.1 to 0.7 seconds) because it also draws Grad-CAM and includes the first-call warmup."
    ))
    story.append(P("Which configuration was kept", "H2"))
    story.append(P(
        "Selection uses validation macro-F1 only. Test numbers below are from the restored best epoch."
    ))
    story.append(grid(
        ["Run", "Best epoch", "Val macro-F1", "Test acc.", "Test macro-F1", "Train time"],
        [
            ["Brain, fine-tune", "6 of 8", "98.69%", "99.08%", "99.03%", "5 min 59 s"],
            ["Brain, frozen", "4 of 6", "87.84%", "82.91%", "82.18%", "2 min 59 s"],
            ["Skin, fine-tune", "6 of 8", "71.66%", "79.40%", "70.78%", "7 min 29 s"],
            ["Skin, frozen", "2 of 5", "34.43%", "59.75%", "34.57%", "2 min 22 s"],
        ],
        [32 * mm, 24 * mm, 28 * mm, 24 * mm, 30 * mm, 32 * mm],
    ))
    story.append(P(
        "Fine-tuning is the selected model. On brain MRI the frozen features already reach the low 80s, and fine-tuning adds about 16 accuracy points. "
        "On skin, the frozen head is below the 66.9% nevus baseline, so ImageNet features alone are not enough for dermoscopy.",
        "Caption",
    ))

    story.append(P("Selected test results", "H2"))
    story.append(grid(
        ["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Inference", "Train time"],
        [
            ["Brain category", "99.08%", "99.07%", "99.00%", "99.03%", "3.8 ms", "5 min 59 s"],
            ["Brain tumor / no tumor", "99.69%", "99.64%", "99.64%", "99.64%", "3.1 ms", "5 min 22 s"],
            ["Breast subtypes", "60.31%", "50.33%", "62.21%", "54.33%", "2.9 ms", "10 min 49 s"],
            ["Breast benign / malignant", "91.80%", "91.87%", "91.49%", "91.65%", "2.8 ms", "10 min 50 s"],
            ["Skin category", "79.40%", "67.12%", "75.70%", "70.78%", "4.1 ms", "7 min 29 s"],
            ["Skin cancer-related", "89.78%", "82.94%", "87.08%", "84.74%", "3.0 ms", "6 min 34 s"],
            ["Lung category", "96.36%", "93.17%", "94.44%", "93.78%", "2.8 ms", "1 min 2 s"],
            ["Lung malignant / not", "99.39%", "99.41%", "99.38%", "99.39%", "2.9 ms", "47 s"],
        ],
        [38 * mm, 20 * mm, 20 * mm, 20 * mm, 22 * mm, 22 * mm, 28 * mm],
    ))
    story.append(P("Inference is one image. Batched time is 1.2 to 1.4 ms per image. Macro P and macro R are precision and recall.", "Caption"))

    story.append(P("What succeeded and what failed on the test sets", "H2"))
    story.append(P(
        "Brain category, 1,311 test images, 12 errors. No-tumor recall is 100% on this folder (405/405). The model did call two meningiomas “no tumor.” "
        "The other errors are glioma or pituitary read as meningioma: neighboring tumor categories. Macro F1 is 99.03%. "
        "Grouping those four classes into tumor versus no tumor is slightly more accurate (1,309/1,311) than the separate binary model (1,307/1,311). "
        "The second model is still in the app because the page has to show both questions, and because its wording is “tumor,” not “cancer.”"
    ))
    story.append(figure("brain_finetune_confusion.png", "Figure 1. Brain category confusion matrix on the official test folder. Rows are the true class."))
    story.append(P(
        "Skin category, 2,005 test images. Accuracy 79.40% beats the 66.9% nevus baseline. Macro F1 is 70.78%, which is the number that matters. "
        "Melanoma precision is 48.4%: when the model says melanoma it is wrong about half the time, mostly on nevi (120) and benign keratosis (32). "
        "Of 223 true melanomas, 152 are recovered and 38 are called nevi. That pair is the failure accuracy hides. "
        "The coarser skin model is better at its own question: 1,800/2,005 (89.78%) against 1,741/2,005 if the seven-class predictions are only grouped afterwards. "
        "Cancer-related precision is still 70.3%, and 68 cancer-related images are missed."
    ))
    story.append(figure("skin_finetune_confusion.png", "Figure 2. Skin category confusion matrix. The large off-diagonal cells are nevus predicted as melanoma or as benign keratosis."))
    story.append(P(
        "Breast subtypes do not travel well across patients. Test accuracy is 60.31% and macro F1 is 54.33%, while validation macro-F1 was only 27.40%. "
        "Phyllodes recall is 0: the model never predicted that class on the test set, and 140 of 235 phyllodes slides were called fibroadenoma. "
        "The benign-versus-malignant model, on the same patients, is the stable result: test macro F1 91.65% against validation macro F1 91.68%. "
        "76 benign slides were called malignant and 50 malignant slides were called benign."
    ))
    story.append(figure("breast_finetune_confusion.png", "Figure 3. Breast subtype confusion matrix. The phyllodes column is empty: that class was never predicted on the test patients."))
    story.append(P(
        "Lung category test accuracy is 96.36% and macro F1 is 93.78%, on 165 slices. No benign or normal test slice was called malignant. "
        "One malignant slice was called normal and one was called benign. Benign F1 sits on only 18 test slices. "
        "The malignant-or-not model is 164/165. The single error is a not-malignant slice called malignant. "
        "I do not treat 99% or a ROC-AUC of 100% as a clinical or patient-level result, because the split can leak scans."
    ))
    story.append(figure("lung_finetune_confusion.png", "Figure 4. Lung category confusion matrix. Mistakes stay between normal and benign, except one malignant slice called normal and one called benign."))

    story.append(P("7. Deployment", "H1"))
    story.append(P(
        "app.py is a Gradio application. The user picks Brain MRI, Breast histopathology, Skin lesion, or Lung CT, uploads an image, and presses Run model. "
        "The page returns both answers: the specific category, with a sentence that says whether that category is a tumor, a malignant lesion, a precancerous lesion, or a benign lesion in the dataset; "
        "then the coarser model’s answer, and whether the two models agree. It also shows both score charts and a Grad-CAM image from the category model. "
        "If the top score is under 70%, or the next score is within 15 points, the text says that model is not sure. "
        "The disclaimer at the top says a tumor is not cancer. Example rows are held-out public images, one per category."
    ))
    story.append(P("Run it from the project folder, after the virtual environment is active: python app.py. The page opens at http://127.0.0.1:7860.", "Small"))

    story.append(P("8. Real images", "H1"))
    story.append(note_box(
        "<b>Note for the teacher.</b> My real MRI pictures are in phone_photos/myMRI.pdf. "
        "They were taken by an MRI machine, not with a phone. "
        "This is a medical project, so a phone photo of a person, a room, or a screen would not be a fair test. "
        "The file is in the repository: https://github.com/FarahBebs/Two-Answers/blob/main/phone_photos/myMRI.pdf. "
        "I kept the same five brain images below, with the true label next to the model’s prediction. "
        "This is not a diagnosis."
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(P(
        "The last two brain images are cases the model got wrong, so the row is not only the easy ones. "
        "Grad-CAM shows which part of the image pushed the winning class. It is not an outline of a tumor, and it can look tidy when the class is still wrong. "
        "Breast, skin, and lung do not come from my own scans. Those figures are held-out images from the public test sets."
    ))

    story.append(P("My MRI photos", "H2"))
    story.append(figure("brain_five_heldout.png", "Figure 5. My real MRI pictures, taken by an MRI machine. The true class is shown next to the model’s class.", path=mri_figure_for_report()))
    story.append(figure("brain_five_gradcam.png", "Figure 6. Grad-CAM for the same five brain images."))
    story.append(grid(
        ["True label", "Prediction", "Score", "Comment"],
        [
            ["Pituitary tumor", "Pituitary tumor", "100%", "Success. Bright focus near the skull base."],
            ["No tumor", "No tumor", "100%", "Success. Heat is spread across vessels, not a mass outline."],
            ["Meningioma", "Meningioma", "100%", "Success. Heat sits on the bright extra-axial mass."],
            ["Pituitary tumor", "Meningioma", "99.1%", "Failure. Same mix-up as the test matrix. The score is still about 99%."],
            ["Pituitary tumor", "Meningioma", "99.1%", "Failure. Confident and wrong. Neighboring tumor categories."],
        ],
        [32 * mm, 32 * mm, 18 * mm, 88 * mm],
    ))
    story.append(Spacer(1, 2 * mm))

    story.append(P("Skin, five held-out public lesions", "H2"))
    story.append(figure("skin_five_heldout.png", "Figure 7. Five skin test images with the true class and the model’s class."))
    story.append(grid(
        ["True label", "Prediction", "Score", "Comment"],
        [
            ["Vascular lesion", "Vascular lesion", "100%", "Success. A distinct purple lesion. Heat sits on it."],
            ["Melanocytic nevus", "Melanocytic nevus", "100%", "Success. A brown nevus."],
            ["Dermatofibroma", "Dermatofibroma", "100%", "Success. A pale lesion. Heat sits on it."],
            ["Actinic keratoses", "Basal cell carcinoma", "99.9%", "Failure. Pink, poorly pigmented lesions look alike here."],
            ["Actinic keratoses", "Basal cell carcinoma", "99.2%", "Failure. Matches the 10 akiec-to-bcc cells in the test matrix."],
        ],
        [34 * mm, 36 * mm, 18 * mm, 82 * mm],
    ))
    story.append(Spacer(1, 2 * mm))

    story.append(P("Breast and lung, five held-out public images each", "H2"))
    story.append(figure("breast_five_heldout.png", "Figure 8. Five BreakHis test slides. Two phyllodes slides are called fibroadenoma at about 100%."))
    story.append(P(
        "The breast row succeeds on a fibroadenoma, an adenosis, and a papillary carcinoma, all near 100%. "
        "It fails on two phyllodes slides, both called fibroadenoma at about 100%. Both classes are benign in this dataset, so the coarser model can still be right when the subtype is wrong. A certain score is not evidence the subtype is right."
    ))
    story.append(figure("lung_five_heldout.png", "Figure 9. Five lung test slices. The two mistakes stay on the not-malignant side."))
    story.append(P(
        "The lung row succeeds on a malignant slice, a normal slice, and a benign slice. "
        "It fails by calling a normal slice benign (about 84%) and a benign slice normal (about 84%). Neither mistake is called malignant."
    ))

    story.append(P("9. Report fields", "H1"))
    story.append(P(
        "The table below is the assignment template. The same architecture, input size, optimizer, activation, loss, metrics, environment, and deployment are shared. "
        "Dataset, URL, and the measured performance differ, so each model has its own best performance, inference time, and training time in section 6. "
        "The repository line is included because the working project is public."
    ))

    common_tail = [
        ("Repository", "https://github.com/FarahBebs/Two-Answers"),
    ]

    blocks = [
        ("Brain MRI, four categories", [
            ("Project Title", "Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images"),
            ("Problem / Task", "Image classification. Four classes: glioma, meningioma, pituitary tumor, no tumor."),
            ("Dataset", "Brain Tumor MRI (Nickparvar; figshare, SARTAJ, Br35H), CC BY 4.0. Official test folder held out. Training folder split 85/15, seed 42."),
            ("Dataset URL", "https://doi.org/10.5281/zenodo.12735702"),
            *SHARED,
            ("Best Performance", "Test accuracy 99.08%, macro precision 99.07%, macro recall 99.00%, macro F1 99.03%. Validation macro-F1 98.69% at epoch 6 of 8."),
            ("Inference Time", "3.8 ms per image. 1.2 ms per image in a batch of 32."),
            ("Training Time", "5 min 59 s."),
            DEPLOY,
            *common_tail,
        ]),
        ("Brain MRI, tumor or no tumor", [
            ("Project Title", "Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images"),
            ("Problem / Task", "Image classification. Two classes: tumor detected, or no tumor detected. This is not a cancer label."),
            ("Dataset", "Same brain images. Glioma, meningioma, and pituitary become “Tumor detected.” The no-tumor folder stays “No tumor detected.”"),
            ("Dataset URL", "https://doi.org/10.5281/zenodo.12735702"),
            *SHARED,
            ("Best Performance", "Test accuracy 99.69%, macro precision 99.64%, macro recall 99.64%, macro F1 99.64%. Epoch 5 of 8."),
            ("Inference Time", "3.1 ms per image. 1.2 ms per image in a batch of 32."),
            ("Training Time", "5 min 22 s."),
            DEPLOY,
            *common_tail,
        ]),
        ("Breast microscopy, eight subtypes", [
            ("Project Title", "Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images"),
            ("Problem / Task", "Image classification. Eight BreakHis subtypes."),
            ("Dataset", "BreakHis, 7,909 slides, 82 patients. Patient-level split, seed 42: 54 / 14 / 14 patients."),
            ("Dataset URL", "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"),
            *SHARED,
            ("Best Performance", "Test accuracy 60.31%, macro precision 50.33%, macro recall 62.21%, macro F1 54.33%. Phyllodes recall is 0. Epoch 5 of 8."),
            ("Inference Time", "2.9 ms per image. 1.4 ms per image in a batch of 32."),
            ("Training Time", "10 min 49 s."),
            DEPLOY,
            *common_tail,
        ]),
        ("Breast microscopy, benign or malignant", [
            ("Project Title", "Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images"),
            ("Problem / Task", "Image classification. Benign or malignant, using BreakHis’s own folders."),
            ("Dataset", "Same BreakHis images and the same patient split. Phyllodes stays in the benign folder of this dataset."),
            ("Dataset URL", "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"),
            *SHARED,
            ("Best Performance", "Test accuracy 91.80%, macro precision 91.87%, macro recall 91.49%, macro F1 91.65%. Validation macro-F1 91.68%. Epoch 6 of 8."),
            ("Inference Time", "2.8 ms per image. 1.3 ms per image in a batch of 32."),
            ("Training Time", "10 min 50 s."),
            DEPLOY,
            *common_tail,
        ]),
        ("Skin lesion, seven categories", [
            ("Project Title", "Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images"),
            ("Problem / Task", "Image classification. Seven HAM10000 categories."),
            ("Dataset", "DermaMNIST 224, from HAM10000, CC BY-NC 4.0. Official MedMNIST split. Not reshuffled."),
            ("Dataset URL", "https://doi.org/10.7910/DVN/DBW86T"),
            *SHARED,
            ("Best Performance", "Test accuracy 79.40%, macro precision 67.12%, macro recall 75.70%, macro F1 70.78%. Melanoma precision 48.4%. Epoch 6 of 8."),
            ("Inference Time", "4.1 ms per image. 1.2 ms per image in a batch of 32."),
            ("Training Time", "7 min 29 s."),
            DEPLOY,
            *common_tail,
        ]),
        ("Skin lesion, cancer-related or not", [
            ("Project Title", "Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images"),
            ("Problem / Task", "Image classification. Cancer-related (melanoma, basal cell carcinoma, actinic keratoses) or not. Actinic keratoses are precancerous, not invasive cancer."),
            ("Dataset", "Same DermaMNIST images and the same official split."),
            ("Dataset URL", "https://doi.org/10.7910/DVN/DBW86T"),
            *SHARED,
            ("Best Performance", "Test accuracy 89.78%, macro precision 82.94%, macro recall 87.08%, macro F1 84.74%. Epoch 6 of 8."),
            ("Inference Time", "3.0 ms per image. 1.2 ms per image in a batch of 32."),
            ("Training Time", "6 min 34 s."),
            DEPLOY,
            *common_tail,
        ]),
        ("Lung CT, three labels", [
            ("Project Title", "Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images"),
            ("Problem / Task", "Image classification of CT slices: normal, benign, or malignant. Not nodule detection."),
            ("Dataset", "IQ-OTH/NCCD, 1,097 slices, CC0. Image-level 70/15/15 split, seed 42. Not patient-level."),
            ("Dataset URL", "https://doi.org/10.17632/bhmdr45bh2.4"),
            *SHARED,
            ("Best Performance", "Test accuracy 96.36%, macro precision 93.17%, macro recall 94.44%, macro F1 93.78%. Epoch 6 of 8."),
            ("Inference Time", "2.8 ms per image. 1.2 ms per image in a batch of 32."),
            ("Training Time", "1 min 2 s."),
            DEPLOY,
            *common_tail,
        ]),
        ("Lung CT, malignant or not", [
            ("Project Title", "Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images"),
            ("Problem / Task", "Image classification. Not malignant (normal and benign together) or malignant."),
            ("Dataset", "Same IQ-OTH/NCCD slices and the same image split. A benign slice is not called cancer."),
            ("Dataset URL", "https://doi.org/10.17632/bhmdr45bh2.4"),
            *SHARED,
            ("Best Performance", "Test accuracy 99.39%, macro F1 99.39%, one error in 165 slices. ROC-AUC 100% on this split. Epoch 3 of 6. Not a patient-level result."),
            ("Inference Time", "2.9 ms per image. 1.2 ms per image in a batch of 32."),
            ("Training Time", "47 s."),
            DEPLOY,
            *common_tail,
        ]),
    ]
    for title, rows in blocks:
        story.append(CondPageBreak(70 * mm))
        story.append(P(title, "H2"))
        story.append(field_table(rows))
        story.append(Spacer(1, 3 * mm))

    story.append(P("What this report does not claim", "H1"))
    story.append(bullets([
        "The app does not detect cancer. The brain yes-or-no model detects a tumor category. The skin yes-or-no model detects a cancer-related HAM10000 label, including a precancerous one.",
        "99% on the brain test folder is real for that folder. It is not a measurement on a new hospital, because patient identity is not in the files.",
        "79% skin accuracy still leaves melanoma precision at 48%. That is not a screening result.",
        "BreakHis subtype accuracy is 60% on a patient split. The 92% benign-versus-malignant number is the one that holds across patients.",
        "The lung scores use an image split. They are not a patient-level result, and they are not nodule detection.",
        "My real MRI pictures are in phone_photos/myMRI.pdf. They were taken by an MRI machine, not with a phone. The breast, skin, and lung figures are still public test images.",
    ]))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="Two Answers: Category and Group Labels for Brain, Breast, Skin, and Lung Images",
        author="Farah Alhasan",
        subject="Computer Vision homework. Deep learning application.",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT)


if __name__ == "__main__":
    build()
