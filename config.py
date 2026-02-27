"""
Centralized configuration for the food recognition pipeline.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "datasets", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ─── Kaggle Datasets ────────────────────────────────────────────────────────
KAGGLE_DATASETS = {
    "indian_food_images": {
        "kaggle_path": "iamsouravbanerjee/indian-food-images-dataset",
        "expected_classes": 80,
        "description": "Indian Food Images (dosa, biryani, samosa, etc.)",
    },
    "mafood121": {
        "kaggle_path": "theviz/mafood121",
        "expected_classes": 121,
        "description": "MAFood-121 Multi-cuisine (Indian, Thai, Chinese, Western)",
    },
    "uecfood256": {
        "kaggle_path": "rkuo2000/uecfood256",
        "expected_classes": 256,
        "description": "UECFood-256 Japanese/Asian food",
    },
    "food101": {
        "kaggle_path": "dansbecker/food-101",
        "expected_classes": 101,
        "description": "Food-101 Western food baseline",
    },
    "massive_indian_food": {
        "kaggle_path": "anshulmehtakaggl/themassiveindianfooddataset",
        "expected_classes": 15,
        "description": "Massive Indian Food Dataset (high-quality staples)",
    },
}

# ─── Image Settings ──────────────────────────────────────────────────────────
IMG_SIZE = 224  # MobileNetV2 default input size
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

# ─── Data Split ──────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
MIN_IMAGES_PER_CLASS = 5  # Classes with fewer images are dropped
RANDOM_SEED = 42

# ─── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 32
PHASE1_EPOCHS = int(os.environ.get("PHASE1_EPOCHS", 10))   # Feature extraction (frozen base); override with env var
PHASE2_EPOCHS = int(os.environ.get("PHASE2_EPOCHS", 20))   # Fine-tuning (top layers unfrozen); override with env var
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
DROPOUT_RATE = 0.2
EARLY_STOPPING_PATIENCE = 5
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5

# ─── TFLite Conversion ──────────────────────────────────────────────────────
TFLITE_FLOAT16_PATH = os.path.join(MODEL_DIR, "food_recognition_float16.tflite")
TFLITE_INT8_PATH = os.path.join(MODEL_DIR, "food_recognition_int8.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "class_mapping.json")
NUM_CALIBRATION_IMAGES = 100  # For int8 quantization
