# 🧠 Brain Tumor Segmentation System

> AI-powered 3D MRI segmentation using a **3D U-Net** trained on the **BraTS** dataset.
> Built with PyTorch, MONAI, and Streamlit.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Training the Model](#training-the-model)
7. [Running the App](#running-the-app)
8. [Email Configuration](#email-configuration)
9. [Limitations](#limitations)

---

## Project Overview

This system provides an end-to-end pipeline for brain tumor segmentation from multi-modal MRI scans:

| Component | Details |
|---|---|
| UI | Streamlit multi-page app |
| Model | 3D U-Net (MONAI) |
| Input | 4 MRI modalities: T1, T1ce, T2, FLAIR |
| Output | Voxel-wise segmentation mask (4 classes) |
| Report | Plain-text clinical report with email delivery |

**Segmentation classes (BraTS convention):**
- Class 0 – Background
- Class 1 – Necrotic Core / Non-enhancing Tumor
- Class 2 – Peritumoral Edema
- Class 3 – Enhancing Tumor (BraTS label 4 remapped to 3)

---

## Architecture

```
3D U-Net (MONAI UNet)
─────────────────────────────────────────────────────────────
Input  : (B, 4, 128, 128, 128)  ← 4 MRI modalities

Encoder:
  Conv Block (16 filters)  → stride-2 down → (16, 64, 64, 64)
  Conv Block (32 filters)  → stride-2 down → (32, 32, 32, 32)
  Conv Block (64 filters)  → stride-2 down → (64, 16, 16, 16)
  Conv Block (128 filters) → stride-2 down → (128, 8, 8, 8)

Bottleneck:
  Conv Block (256 filters)                 → (256, 8, 8, 8)

Decoder (with skip connections):
  UpConv + Cat → (128, 16, 16, 16)
  UpConv + Cat → (64, 32, 32, 32)
  UpConv + Cat → (32, 64, 64, 64)
  UpConv + Cat → (16, 128, 128, 128)

Output : (B, 4, 128, 128, 128)  ← class logits
─────────────────────────────────────────────────────────────
Parameters : ~31 M
Norm       : Batch Normalisation
Residual   : 2 residual units per block
Dropout    : 0.1
```

**Loss function:** Combined Dice + Cross-Entropy  
`L = 0.4 × CE + 0.6 × (1 − DiceLoss)`

**Optimiser:** Adam | lr=0.001 | weight_decay=1e-5  
**Scheduler:** Cosine Annealing (T_max=epochs, η_min=1e-6)

---

## Dataset

**BraTS 2020** – Brain Tumor Segmentation Challenge  
Download: https://www.med.upenn.edu/cbica/brats2020/

**Expected folder structure:**
```
BraTS2020_TrainingData/
  BraTS20_Training_001/
    BraTS20_Training_001_t1.nii.gz
    BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_seg.nii.gz
  BraTS20_Training_002/
    ...
```

The dataset loader also supports a simpler layout where files are named
`t1.nii`, `t1ce.nii`, `t2.nii`, `flair.nii`, `seg.nii` inside each sub-folder.

---

## Project Structure

```
project/
├── main.py                       ← Streamlit entry point
├── train.py                      ← Training script (run this first)
├── train_colab.ipynb             ← Google Colab training notebook
├── requirements.txt
├── .env.example                  ← Copy to .env and fill in credentials
│
├── src/
│   ├── dataset.py                ← BraTSDataset + DataLoader factory
│   ├── model_loader.py           ← Model loading, inference
│   ├── preprocess.py             ← N4 correction, skull stripping, normalisation
│   ├── utils.py                  ← Metrics (Dice, IoU), report helpers
│   ├── visualize.py              ← Overlay & slice visualisation
│   ├── email_handler.py          ← SMTP email with attachment
│   └── assets.py                 ← UI assets
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Upload_MRI.py
│   ├── 3_Run_Segmentation.py     ← Upgraded: auto-detects real vs mock model
│   ├── 4_View_Results.py
│   ├── 5_Generate_Report.py      ← Upgraded: email with attachment
│   ├── 6_History.py
│   └── 7_System_Info.py
│
└── checkpoints/                  ← Created by train.py
    ├── model.pth                 ← Best model (used by app)
    ├── best_model.pth
    ├── last_model.pth
    └── training_log.json
```

---

## Installation

```bash
# 1. Clone / unzip the project
cd project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Set up .env
cp .env.example .env
# Edit .env with your Gmail App Password
```

---

## Training the Model

### Option A – Local (GPU recommended)

```bash
python train.py \
    --data_dir      /path/to/BraTS2020_TrainingData \
    --checkpoint_dir checkpoints \
    --epochs         50 \
    --batch_size     2 \
    --lr             0.001 \
    --val_split      0.2 \
    --input_shape    128 128 128

# After training, checkpoints/model.pth is created automatically.
```

### Option B – Google Colab (free GPU)

1. Open `train_colab.ipynb` in Colab
2. Set Runtime → Change runtime type → **GPU**
3. Follow the cells step by step
4. Download `model.pth` at the end and place it in `checkpoints/`

**Training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | (required) | Path to BraTS root |
| `--checkpoint_dir` | `checkpoints` | Where to save .pth files |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 2 | Batch size (reduce to 1 for 8 GB VRAM) |
| `--lr` | 0.001 | Adam learning rate |
| `--val_split` | 0.2 | Validation fraction |
| `--input_shape` | 128 128 128 | Resize target |
| `--resume` | None | Path to checkpoint to continue from |

---

## Running the App

```bash
# Make sure you are in the project root
streamlit run main.py
```

Then open http://localhost:8501 in your browser.

**App will automatically detect** `checkpoints/model.pth` and use the real
trained model. If no checkpoint is found it falls back to the mock model
with a visible warning.

---

## Email Configuration

1. Enable 2-Step Verification on your Google account
2. Go to: Google Account → Security → App passwords
3. Generate a new App Password (16 characters)
4. Add to `.env`:

```
SENDER_EMAIL=your.email@gmail.com
SENDER_PASSWORD=abcd efgh ijkl mnop
```

The report will be attached as a `.txt` file to the email.

---

## Limitations

- **Not a medical device.** All results are for research / educational purposes only.
- The mock model generates synthetic predictions using intensity thresholding — not suitable for clinical use.
- 128³ input crops may lose detail from larger scans (use sliding-window inference for full-resolution).
- Preprocessing (N4 bias correction, skull stripping) uses simplified implementations; clinical pipelines use FSL/FreeSurfer.
- Training on the full BraTS 2020 dataset (~369 cases) on a single GPU takes ~6–12 hours.

---

## References

- Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)", IEEE TMI 2015
- MONAI: https://monai.io
- BraTS Challenge: https://www.med.upenn.edu/cbica/brats2020/
