#!/usr/bin/env python3
"""
Generate a sample food dataset for pipeline testing.

This creates synthetic images with distinct visual patterns for each food class,
allowing the full training pipeline to be tested without Kaggle downloads.

The model trained on this data won't recognize real food, but it proves the
entire pipeline (train → evaluate → TFLite convert → inference) works correctly.

Replace this data with real Kaggle datasets for a production model.

Usage:
    python generate_sample_data.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, PROCESSED_DATA_DIR

# 30 food classes to simulate - these match real classes from the Kaggle datasets
SAMPLE_CLASSES = [
    "biryani", "butter_chicken", "chicken_curry", "chole_bhature",
    "dal_makhani", "dosa", "fried_rice", "gulab_jamun",
    "idli", "jalebi", "masala_dosa", "naan",
    "palak_paneer", "pav_bhaji", "samosa", "tandoori_chicken",
    "chapati", "rasgulla", "vada_pav", "pani_puri",
    "hamburger", "pizza", "sushi", "ramen",
    "pad_thai", "spring_roll", "ice_cream", "chocolate_cake",
    "french_fries", "pancakes",
]

# Unique color + pattern per class so the model can learn real visual distinctions
CLASS_VISUALS = {}
np.random.seed(42)
for i, cls in enumerate(SAMPLE_CLASSES):
    hue = int((i / len(SAMPLE_CLASSES)) * 360)
    # Convert HSV-like to RGB (simple approach)
    r = int(128 + 127 * np.sin(np.radians(hue)))
    g = int(128 + 127 * np.sin(np.radians(hue + 120)))
    b = int(128 + 127 * np.sin(np.radians(hue + 240)))
    CLASS_VISUALS[cls] = {
        "base_color": (r, g, b),
        "pattern_seed": i * 7 + 13,
    }

IMAGES_PER_CLASS_TRAIN = 80
IMAGES_PER_CLASS_VAL = 10
IMAGES_PER_CLASS_TEST = 10


def generate_image(class_name, variation_seed):
    """Generate a synthetic image with class-specific visual patterns."""
    vis = CLASS_VISUALS[class_name]
    rng = np.random.RandomState(vis["pattern_seed"] + variation_seed)

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), vis["base_color"])
    draw = ImageDraw.Draw(img)
    arr = np.array(img, dtype=np.float32)

    # Add class-specific texture patterns
    noise = rng.randn(IMG_SIZE, IMG_SIZE, 3) * 30
    arr = np.clip(arr + noise, 0, 255)

    # Add geometric shapes unique to each class
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    pattern_type = vis["pattern_seed"] % 5
    color2 = tuple(rng.randint(50, 255, 3).tolist())

    if pattern_type == 0:  # Circles
        for _ in range(rng.randint(3, 8)):
            x, y = rng.randint(20, IMG_SIZE - 20, 2)
            r = rng.randint(10, 40)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color2)
    elif pattern_type == 1:  # Rectangles
        for _ in range(rng.randint(2, 6)):
            x1, y1 = rng.randint(10, IMG_SIZE // 2, 2)
            x2, y2 = x1 + rng.randint(20, 80), y1 + rng.randint(20, 80)
            draw.rectangle([x1, y1, x2, y2], fill=color2)
    elif pattern_type == 2:  # Lines
        for _ in range(rng.randint(5, 15)):
            pts = rng.randint(0, IMG_SIZE, 4).tolist()
            draw.line(pts, fill=color2, width=rng.randint(2, 5))
    elif pattern_type == 3:  # Triangles
        for _ in range(rng.randint(2, 5)):
            pts = rng.randint(10, IMG_SIZE - 10, 6).tolist()
            draw.polygon(pts, fill=color2)
    else:  # Gradient stripes
        stripe_w = rng.randint(10, 30)
        for x in range(0, IMG_SIZE, stripe_w * 2):
            draw.rectangle([x, 0, x + stripe_w, IMG_SIZE], fill=color2)

    # Add slight per-image variation (brightness, crop offset)
    arr = np.array(img, dtype=np.float32)
    brightness = rng.uniform(0.8, 1.2)
    arr = np.clip(arr * brightness, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def generate_split(split_dir, images_per_class):
    """Generate images for one split (train/val/test)."""
    total = 0
    for cls in SAMPLE_CLASSES:
        cls_dir = os.path.join(split_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(images_per_class):
            img = generate_image(cls, variation_seed=i * 1000 + hash(split_dir) % 1000)
            img.save(os.path.join(cls_dir, f"{cls}_{i:04d}.jpg"), "JPEG", quality=90)
            total += 1

    return total


def main():
    print("=" * 70)
    print("GENERATING SAMPLE FOOD DATASET")
    print("=" * 70)
    print(f"\nClasses: {len(SAMPLE_CLASSES)}")
    print(f"Images per class: train={IMAGES_PER_CLASS_TRAIN}, "
          f"val={IMAGES_PER_CLASS_VAL}, test={IMAGES_PER_CLASS_TEST}")
    print()

    # Generate train split
    print("Generating training images...")
    n = generate_split(TRAIN_DIR, IMAGES_PER_CLASS_TRAIN)
    print(f"  Created {n} training images")

    # Generate val split
    print("Generating validation images...")
    n = generate_split(VAL_DIR, IMAGES_PER_CLASS_VAL)
    print(f"  Created {n} validation images")

    # Generate test split
    print("Generating test images...")
    n = generate_split(TEST_DIR, IMAGES_PER_CLASS_TEST)
    print(f"  Created {n} test images")

    # Generate labels.txt and class_mapping.json
    import json
    from config import LABELS_PATH, CLASS_MAPPING_PATH

    class_to_id = {cls.replace("_", " "): i for i, cls in enumerate(sorted(SAMPLE_CLASSES))}
    with open(CLASS_MAPPING_PATH, "w") as f:
        json.dump(class_to_id, f, indent=2, sort_keys=True)

    id_to_class = {v: k for k, v in class_to_id.items()}
    with open(LABELS_PATH, "w") as f:
        for i in range(len(id_to_class)):
            f.write(id_to_class[i] + "\n")

    total_images = len(SAMPLE_CLASSES) * (
        IMAGES_PER_CLASS_TRAIN + IMAGES_PER_CLASS_VAL + IMAGES_PER_CLASS_TEST
    )
    print(f"\nTotal: {total_images} images across {len(SAMPLE_CLASSES)} classes")
    print(f"Labels: {LABELS_PATH}")
    print(f"Class mapping: {CLASS_MAPPING_PATH}")
    print(f"\nData ready at: {PROCESSED_DATA_DIR}")
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
