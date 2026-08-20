# U-Net for Carvana Image Masking

A PyTorch implementation of **U-Net** for the [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) — binary semantic segmentation that separates a car from its background with pixel-level precision.

<img width="914" height="615" alt="Screenshot 2026-08-20 at 2 51 43 PM" src="https://github.com/user-attachments/assets/7c224a27-24c5-43b8-9753-dd2b7390c0f7" />

---

## 📌 Overview

U-Net is a fully convolutional encoder-decoder network originally built for biomedical image segmentation. Its symmetric contracting/expanding path with **skip connections** makes it exceptionally good at producing precise, high-resolution masks — which is exactly what the Carvana challenge needs: crisp car-vs-background masks from studio photos.

This repo trains a U-Net from scratch on the Carvana dataset to predict a binary mask for every pixel in an image.

---

## 🧠 Architecture

U-Net consists of a **contracting path** (encoder) that captures context, and a symmetric **expanding path** (decoder) that enables precise localization. Skip connections concatenate feature maps from the encoder directly into the corresponding decoder stage, preserving fine spatial detail lost during downsampling.

```mermaid
flowchart TB
    subgraph Encoder["Contracting Path (Encoder)"]
        direction TB
        I["Input Image<br/>3 x 572 x 572"] --> C1["Conv 3x3, ReLU x2<br/>64 filters"]
        C1 --> P1["MaxPool 2x2"]
        P1 --> C2["Conv 3x3, ReLU x2<br/>128 filters"]
        C2 --> P2["MaxPool 2x2"]
        P2 --> C3["Conv 3x3, ReLU x2<br/>256 filters"]
        C3 --> P3["MaxPool 2x2"]
        P3 --> C4["Conv 3x3, ReLU x2<br/>512 filters"]
        P4["MaxPool 2x2"]
        C4 --> P4
    end

    P4 --> BN["Bottleneck<br/>Conv 3x3, ReLU x2<br/>1024 filters"]

    subgraph Decoder["Expanding Path (Decoder)"]
        direction TB
        BN --> U4["UpConv 2x2<br/>512 filters"]
        U4 --> D4["Conv 3x3, ReLU x2<br/>512 filters"]
        D4 --> U3["UpConv 2x2<br/>256 filters"]
        U3 --> D3["Conv 3x3, ReLU x2<br/>256 filters"]
        D3 --> U2["UpConv 2x2<br/>128 filters"]
        U2 --> D2["Conv 3x3, ReLU x2<br/>128 filters"]
        D2 --> U1["UpConv 2x2<br/>64 filters"]
        U1 --> D1["Conv 3x3, ReLU x2<br/>64 filters"]
        D1 --> OUT["Conv 1x1<br/>1 x H x W (mask)"]
    end

    C4 -. skip connection .-> D4
    C3 -. skip connection .-> D3
    C2 -. skip connection .-> D2
    C1 -. skip connection .-> D1

    style Encoder fill:#1a1a2e,color:#fff
    style Decoder fill:#16213e,color:#fff
    style BN fill:#0f3460,color:#fff
```

**Key design choices in this implementation:**
- `DoubleConv` blocks: `Conv2d → BatchNorm2d → ReLU`, repeated twice
- Downsampling via `MaxPool2d(2)`, upsampling via `ConvTranspose2d` (or bilinear + conv)
- Skip connections concatenated channel-wise before each decoder `DoubleConv`
- Final `1x1` convolution maps to a single-channel logit map (binary mask)
- Loss: `BCEWithLogitsLoss` + Dice Loss (combined) for better boundary accuracy
- Output activation: `Sigmoid` at inference to get a `[0, 1]` probability mask

---

## 📂 Project Structure

```
unet-carvana/
├── data/
│   ├── train_images/
│   ├── train_masks/
│   ├── val_images/
│   └── val_masks/
├── model.py            # U-Net architecture (DoubleConv, Down, Up, OutConv)
├── dataset.py           # CarvanaDataset (torch Dataset + transforms)
├── train.py              # Training loop, checkpointing, validation
├── utils.py             # Dice score, accuracy, save/load checkpoint, save predictions
├── predict.py            # Run inference on new images
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pillow
albumentations
tqdm
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset

1. Download the dataset from the [Kaggle Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data).
2. Unzip and arrange it as:

```
data/
├── train_images/   # .jpg car images
├── train_masks/    # .gif binary masks
```

3. The `dataset.py` script handles a train/val split — adjust the `split ratio` and image `resize` dimensions there as needed.

---

## 🚀 Usage

### Train

```bash
python train.py \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --image-height 160 \
  --image-width 240
```

Checkpoints are saved after every epoch to `checkpoints/`, and the best model (by validation Dice score) is saved separately.

### Predict / Run Inference

```bash
python predict.py --checkpoint checkpoints/best_model.pth --image path/to/car.jpg --output out_mask.png
```

### Resume Training

```bash
python train.py --load-checkpoint checkpoints/best_model.pth
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Validation Dice Score | ~0.99 |
| Validation Pixel Accuracy | ~99.5% |
| Training Time (1x GPU) | ~X min/epoch |

*(Fill in with your actual numbers after training.)*

**Sample prediction:**

```
Input Image  →  Predicted Mask  →  Overlay
   [car.jpg] →   [car_mask.png] →  [car_overlay.png]
```

---

## 🔧 Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Loss Function | BCEWithLogitsLoss (+ Dice) |
| Image Size | 160 × 240 (downscaled for speed) |
| Epochs | 20 |

---

## 📚 References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- [Carvana Image Masking Challenge — Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge)

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
