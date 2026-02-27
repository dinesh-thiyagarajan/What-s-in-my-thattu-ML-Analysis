#!/usr/bin/env python3
"""
Merge, deduplicate, and split all downloaded datasets into a unified structure.

This script:
1. Scans each downloaded dataset to discover class folders and images
2. Normalizes class names across datasets (handles naming inconsistencies)
3. Merges overlapping classes from different datasets
4. Creates a unified train/val/test split
5. Generates class_mapping.json and labels.txt

Usage:
    python prepare_data.py
"""
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    CLASS_MAPPING_PATH,
    LABELS_PATH,
    MIN_IMAGES_PER_CLASS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SUPPORTED_EXTENSIONS,
    TEST_DIR,
    TEST_SPLIT,
    TRAIN_DIR,
    TRAIN_SPLIT,
    VAL_DIR,
    VAL_SPLIT,
)


# ─── Class Name Normalization ───────────────────────────────────────────────
# Manual mapping for known duplicates/variants across datasets
ALIAS_MAP = {
    # Indian food aliases
    "chiken curry": "chicken curry",
    "chicken_curry": "chicken curry",
    "butter_chicken": "butter chicken",
    "dal_makhani": "dal makhani",
    "dal makhni": "dal makhani",
    "naan_bread": "naan",
    "naan bread": "naan",
    "garlic_naan": "garlic naan",
    "tandoori_chicken": "tandoori chicken",
    "palak_paneer": "palak paneer",
    "paneer_butter_masala": "paneer butter masala",
    "chole_bhature": "chole bhature",
    "aloo_gobi": "aloo gobi",
    "aloo gobi ": "aloo gobi",
    "gulab_jamun": "gulab jamun",
    "ras_malai": "ras malai",
    "rasgulla": "rasgulla",
    "jalebi": "jalebi",
    "pav_bhaji": "pav bhaji",
    "masala_dosa": "masala dosa",
    "plain dosa": "dosa",
    "biryani_rice": "biryani",
    "hyderabadi biryani": "biryani",
    "chicken_biryani": "chicken biryani",
    "veg biryani": "vegetable biryani",
    "veg_biryani": "vegetable biryani",
    # Western food aliases
    "french_fries": "french fries",
    "french fry": "french fries",
    "fries": "french fries",
    "hot_dog": "hot dog",
    "hotdog": "hot dog",
    "ice_cream": "ice cream",
    "icecream": "ice cream",
    "grilled_cheese_sandwich": "grilled cheese sandwich",
    "grilled cheese": "grilled cheese sandwich",
    "hamburger": "hamburger",
    "burger": "hamburger",
    "cheese_burger": "cheeseburger",
    "cheese burger": "cheeseburger",
    "caesar_salad": "caesar salad",
    "caesar salad": "caesar salad",
    # Asian food aliases
    "fried_rice": "fried rice",
    "friedrice": "fried rice",
    "spring_roll": "spring roll",
    "spring_rolls": "spring roll",
    "spring rolls": "spring roll",
    "pad_thai": "pad thai",
    "ramen_noodles": "ramen",
    "ramen noodle": "ramen",
    "miso_soup": "miso soup",
    "sushi_roll": "sushi",
    "sashimi_fish": "sashimi",
    "dim_sum": "dim sum",
    "dumpling": "dumplings",
    "gyoza": "dumplings",
    # General
    "chocolate_cake": "chocolate cake",
    "cheese_cake": "cheesecake",
    "cheese cake": "cheesecake",
    "apple_pie": "apple pie",
    "cup_cake": "cupcake",
    "cup cake": "cupcake",
    "pancake": "pancakes",
    "pancake_": "pancakes",
    "waffle": "waffles",
    "omelette": "omelette",
    "omelet": "omelette",
}


def normalize_class_name(name):
    """
    Normalize a class/folder name into a consistent format.

    Steps:
    1. Strip leading/trailing whitespace
    2. Remove numeric prefixes (e.g., "001_" from UECFood)
    3. Replace underscores and hyphens with spaces
    4. Lowercase everything
    5. Apply alias mapping for known duplicates
    6. Strip extra whitespace
    """
    name = name.strip()

    # Remove leading numeric prefix (e.g., "001_rice" → "rice", "23" with only number → keep as-is)
    name = re.sub(r"^\d+[_\-\s]+", "", name)

    # Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")

    # Lowercase
    name = name.lower().strip()

    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)

    # Apply alias mapping
    if name in ALIAS_MAP:
        name = ALIAS_MAP[name]

    return name


def is_valid_image(filepath):
    """Check if a file is a valid, non-corrupted image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def scan_dataset(dataset_path, dataset_name):
    """
    Scan a dataset directory and return a mapping of class_name → [image_paths].

    Handles different dataset structures:
    - Flat: dataset/class1/img.jpg
    - Nested: dataset/subdir/class1/img.jpg (common with Kaggle extractions)
    """
    classes = defaultdict(list)
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"  [WARNING] Dataset path not found: {dataset_path}")
        return classes

    # Find all image files
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(dataset_path.rglob(f"*{ext}"))
        image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))

    if not image_files:
        print(f"  [WARNING] No images found in {dataset_path}")
        return classes

    # Group images by their parent directory (class folder)
    for img_path in image_files:
        # The class name is the immediate parent directory
        class_folder = img_path.parent.name

        # Skip if the parent is the dataset root itself (uncategorized images)
        if img_path.parent == dataset_path:
            continue

        raw_name = class_folder
        normalized = normalize_class_name(raw_name)

        # Skip empty or purely numeric class names (likely not real classes)
        if not normalized or normalized.isdigit():
            # For UECFood-256, numeric folders ARE the classes
            # Try to keep them with a prefix
            if dataset_name == "uecfood256" and raw_name.isdigit():
                normalized = f"uecfood class {raw_name}"
            else:
                continue

        classes[normalized].append(str(img_path))

    return classes


def scan_all_datasets():
    """Scan all downloaded datasets and build a unified class→images mapping."""
    print("Scanning datasets...")
    print()

    unified_classes = defaultdict(list)
    dataset_stats = {}

    for name in sorted(os.listdir(RAW_DATA_DIR)):
        dataset_path = os.path.join(RAW_DATA_DIR, name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"  Scanning: {name}")
        classes = scan_dataset(dataset_path, name)

        # Merge into unified mapping
        for class_name, image_paths in classes.items():
            unified_classes[class_name].extend(image_paths)

        num_classes = len(classes)
        num_images = sum(len(imgs) for imgs in classes.values())
        dataset_stats[name] = {"classes": num_classes, "images": num_images}
        print(f"    Found {num_classes} classes, {num_images} images")

    print()
    return unified_classes, dataset_stats


def filter_classes(unified_classes):
    """Remove classes with too few images."""
    filtered = {}
    dropped = []

    for class_name, image_paths in sorted(unified_classes.items()):
        if len(image_paths) >= MIN_IMAGES_PER_CLASS:
            filtered[class_name] = image_paths
        else:
            dropped.append((class_name, len(image_paths)))

    if dropped:
        print(f"Dropped {len(dropped)} classes with < {MIN_IMAGES_PER_CLASS} images:")
        for name, count in dropped[:20]:  # Show first 20
            print(f"  - {name}: {count} images")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped) - 20} more")
        print()

    return filtered


def create_splits(unified_classes):
    """
    Split each class into train/val/test sets using stratified splitting.

    Returns dict: {split_name: {class_name: [image_paths]}}
    """
    splits = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

    for class_name, image_paths in tqdm(unified_classes.items(), desc="Splitting"):
        images = list(image_paths)
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(images)

        n = len(images)
        if n < 3:
            # Too few for proper split, put all in train
            splits["train"][class_name] = images
            continue

        # First split: separate test set
        val_test_size = VAL_SPLIT + TEST_SPLIT
        try:
            train_imgs, val_test_imgs = train_test_split(
                images, test_size=val_test_size, random_state=RANDOM_SEED
            )
        except ValueError:
            # If not enough samples for the split ratio, do manual split
            n_test = max(1, int(n * TEST_SPLIT))
            n_val = max(1, int(n * VAL_SPLIT))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = 1
                n_val = max(1, (n - 1) // 2)
                n_test = n - 1 - n_val
            train_imgs = images[:n_train]
            val_imgs = images[n_train : n_train + n_val]
            test_imgs = images[n_train + n_val :]
            splits["train"][class_name] = train_imgs
            splits["val"][class_name] = val_imgs
            splits["test"][class_name] = test_imgs
            continue

        # Second split: separate val from test
        relative_test_size = TEST_SPLIT / val_test_size
        try:
            val_imgs, test_imgs = train_test_split(
                val_test_imgs, test_size=relative_test_size, random_state=RANDOM_SEED
            )
        except ValueError:
            mid = len(val_test_imgs) // 2
            val_imgs = val_test_imgs[:mid] if mid > 0 else val_test_imgs[:1]
            test_imgs = val_test_imgs[mid:] if mid > 0 else val_test_imgs[1:]

        splits["train"][class_name] = train_imgs
        splits["val"][class_name] = val_imgs
        splits["test"][class_name] = test_imgs

    return splits


def copy_images(splits, class_to_id):
    """Copy images into the unified directory structure."""
    split_dirs = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}

    # Clean existing processed data
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Removing existing processed data at {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)

    total_copied = 0
    total_skipped = 0

    for split_name, classes in splits.items():
        split_dir = split_dirs[split_name]
        split_count = 0

        for class_name, image_paths in tqdm(
            sorted(classes.items()), desc=f"Copying {split_name}"
        ):
            # Use class name as folder name (replace spaces with underscores for filesystem)
            folder_name = class_name.replace(" ", "_")
            class_dir = os.path.join(split_dir, folder_name)
            os.makedirs(class_dir, exist_ok=True)

            for i, src_path in enumerate(image_paths):
                ext = Path(src_path).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    ext = ".jpg"
                dst_name = f"{folder_name}_{i:05d}{ext}"
                dst_path = os.path.join(class_dir, dst_name)

                try:
                    shutil.copy2(src_path, dst_path)
                    split_count += 1
                    total_copied += 1
                except Exception as e:
                    total_skipped += 1

        print(f"  {split_name}: {split_count} images copied")

    print(f"\nTotal: {total_copied} copied, {total_skipped} skipped")
    return total_copied


def save_class_mapping(class_to_id):
    """Save the class name → ID mapping to JSON."""
    with open(CLASS_MAPPING_PATH, "w") as f:
        json.dump(class_to_id, f, indent=2, sort_keys=True)
    print(f"Class mapping saved to {CLASS_MAPPING_PATH}")


def save_labels(class_to_id):
    """Save labels.txt for Android (one class name per line, ordered by ID)."""
    id_to_class = {v: k for k, v in class_to_id.items()}
    with open(LABELS_PATH, "w") as f:
        for i in range(len(id_to_class)):
            f.write(id_to_class[i] + "\n")
    print(f"Labels file saved to {LABELS_PATH}")


def print_statistics(unified_classes, splits):
    """Print detailed statistics about the dataset."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    total_classes = len(unified_classes)
    total_images = sum(len(imgs) for imgs in unified_classes.values())
    images_per_class = [len(imgs) for imgs in unified_classes.values()]

    print(f"\nTotal classes: {total_classes}")
    print(f"Total images: {total_images}")
    print(f"Images per class - min: {min(images_per_class)}, "
          f"max: {max(images_per_class)}, "
          f"mean: {np.mean(images_per_class):.1f}, "
          f"median: {np.median(images_per_class):.1f}")

    print(f"\nSplit sizes:")
    for split_name, classes in splits.items():
        n_images = sum(len(imgs) for imgs in classes.values())
        n_classes = len(classes)
        print(f"  {split_name}: {n_images} images across {n_classes} classes")

    # Top 20 largest classes
    print(f"\nTop 20 largest classes:")
    sorted_classes = sorted(unified_classes.items(), key=lambda x: len(x[1]), reverse=True)
    for name, imgs in sorted_classes[:20]:
        print(f"  {name}: {len(imgs)} images")

    # Bottom 10 smallest classes (after filtering)
    print(f"\nSmallest 10 classes:")
    for name, imgs in sorted_classes[-10:]:
        print(f"  {name}: {len(imgs)} images")


def main():
    print("=" * 70)
    print("FOOD RECOGNITION - DATA PREPARATION")
    print("=" * 70)
    print()

    # Check raw data exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory not found: {RAW_DATA_DIR}")
        print("Run download_datasets.py first.")
        sys.exit(1)

    # Step 1: Scan all datasets
    unified_classes, dataset_stats = scan_all_datasets()
    print(f"Total unified classes (before filtering): {len(unified_classes)}")
    total = sum(len(v) for v in unified_classes.values())
    print(f"Total images: {total}")
    print()

    # Step 2: Filter classes with too few images
    unified_classes = filter_classes(unified_classes)
    print(f"Total classes (after filtering): {len(unified_classes)}")
    print()

    # Step 3: Create class → ID mapping
    sorted_class_names = sorted(unified_classes.keys())
    class_to_id = {name: i for i, name in enumerate(sorted_class_names)}

    # Step 4: Split into train/val/test
    print("Creating train/val/test splits...")
    splits = create_splits(unified_classes)
    print()

    # Step 5: Copy images to unified structure
    print("Copying images to unified directory structure...")
    copy_images(splits, class_to_id)
    print()

    # Step 6: Save mappings
    save_class_mapping(class_to_id)
    save_labels(class_to_id)
    print()

    # Step 7: Print statistics
    print_statistics(unified_classes, splits)

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Val:   {VAL_DIR}")
    print(f"  Test:  {TEST_DIR}")
    print(f"  Labels: {LABELS_PATH}")
    print(f"  Class mapping: {CLASS_MAPPING_PATH}")
    print("=" * 70)
    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()
