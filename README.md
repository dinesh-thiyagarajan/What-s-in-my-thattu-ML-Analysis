# What's in My Thattu? - Food Recognition ML Pipeline

A complete ML pipeline to train a food recognition model using 5 Kaggle datasets and export it as a **TFLite model for Android**. Pass an image of food → get the food item name.

## Datasets Used

| Dataset | Kaggle Path | Classes | Cuisine |
|---------|------------|---------|---------|
| Indian Food Images | `iamsouravbanerjee/indian-food-images-dataset` | 80 | Indian (dosa, biryani, samosa, etc.) |
| MAFood-121 | `theviz/mafood121` | 121 | Multi-cuisine (Indian, Thai, Chinese, Western) |
| UECFood-256 | `rkuo2000/uecfood256` | 256 | Japanese/Asian |
| Food-101 | `dansbecker/food-101` | 101 | Western food baseline |
| Massive Indian Food | `anshulmehtakaggl/themassiveindianfooddataset` | 15 | High-quality Indian staples |

After merging and deduplication: **~400-500 unique food classes**.

## Model Architecture

- **Base model**: MobileNetV2 (pretrained on ImageNet)
- **Training**: 2-phase transfer learning
  - Phase 1: Feature extraction (frozen base, 10 epochs)
  - Phase 2: Fine-tuning (top 30% unfrozen, 20 epochs)
- **Input**: 224x224 RGB image
- **Output**: Softmax probabilities for each food class
- **TFLite variants**:
  - Float16 quantized (~7 MB, recommended)
  - Int8 quantized (~3.5 MB, for CPU-only devices)

## Quick Start

### 1. Setup

```bash
# Clone the repo
git clone <repo-url>
cd What-s-in-my-thattu-ML-Analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Kaggle API

You need Kaggle API credentials to download datasets:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Under **API** section, click **Create New Token**
3. This downloads `kaggle.json`
4. Place it in the right location:

```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Or use environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 3. Download Datasets

```bash
python download_datasets.py
```

**Alternative**: Download datasets manually from Kaggle and extract them to:
```
datasets/raw/indian_food_images/
datasets/raw/mafood121/
datasets/raw/uecfood256/
datasets/raw/food101/
datasets/raw/massive_indian_food/
```

### 4. Prepare Data

Merges all datasets, normalizes class names, handles duplicates, and creates train/val/test splits:

```bash
python prepare_data.py
```

### 5. Train Model

```bash
python train.py
```

Training uses 2-phase transfer learning with early stopping. On a GPU, this takes approximately 1-3 hours depending on dataset size.

### 6. Evaluate

```bash
python evaluate.py
```

Generates top-1/top-5 accuracy, per-class metrics, and confusion matrix.

### 7. Convert to TFLite

```bash
python convert_tflite.py
```

Outputs:
- `models/food_recognition_float16.tflite` (recommended for Android)
- `models/food_recognition_int8.tflite` (smaller, CPU-optimized)
- `labels.txt` (class names, one per line)

### 8. Test Inference

```bash
# Test with random test images
python test_inference.py

# Test with a specific image
python test_inference.py --image path/to/food.jpg

# Test int8 model
python test_inference.py --model int8
```

## Android Integration

Copy these files to your Android app's `assets/` folder:
- `models/food_recognition_float16.tflite`
- `labels.txt`

### Kotlin Example

```kotlin
// Load TFLite model
val model = Interpreter(loadModelFile("food_recognition_float16.tflite"))

// Preprocess image (224x224 RGB)
val inputBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
// ... load and resize bitmap into buffer

// Run inference
val output = Array(1) { FloatArray(numClasses) }
model.run(inputBuffer, output)

// Get top prediction
val maxIndex = output[0].indices.maxByOrNull { output[0][it] }!!
val foodName = labels[maxIndex]
val confidence = output[0][maxIndex]
```

## Project Structure

```
├── config.py              # Centralized configuration
├── download_datasets.py   # Kaggle dataset downloader
├── prepare_data.py        # Data merging and preprocessing
├── train.py               # Model training (2-phase transfer learning)
├── evaluate.py            # Model evaluation and metrics
├── convert_tflite.py      # TFLite conversion + quantization
├── test_inference.py      # TFLite inference testing
├── requirements.txt       # Python dependencies
├── labels.txt             # Class labels for Android (generated)
├── class_mapping.json     # Class name → ID mapping (generated)
└── README.md              # This file
```

## Configuration

All hyperparameters and paths are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 32 | Training batch size |
| `PHASE1_EPOCHS` | 10 | Feature extraction epochs |
| `PHASE2_EPOCHS` | 20 | Fine-tuning epochs |
| `PHASE1_LR` | 1e-3 | Phase 1 learning rate |
| `PHASE2_LR` | 1e-4 | Phase 2 learning rate |
| `MIN_IMAGES_PER_CLASS` | 5 | Minimum images to keep a class |
| `DROPOUT_RATE` | 0.2 | Dropout rate |

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- ~20 GB disk space for datasets
- GPU recommended for training (CPU works but is slow)
