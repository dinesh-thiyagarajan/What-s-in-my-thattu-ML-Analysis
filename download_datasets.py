#!/usr/bin/env python3
"""
Download all Kaggle datasets for food recognition training.

Usage:
    python download_datasets.py

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Get your Kaggle API token:
       - Go to https://www.kaggle.com/settings
       - Click "Create New Token" under the API section
       - This downloads a kaggle.json file
    3. Place kaggle.json:
       - Linux/Mac: ~/.kaggle/kaggle.json
       - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
    4. Set permissions (Linux/Mac): chmod 600 ~/.kaggle/kaggle.json

Alternatively, set environment variables:
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
"""
import os
import sys
import zipfile
import shutil
from pathlib import Path

from config import KAGGLE_DATASETS, RAW_DATA_DIR


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    env_configured = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")

    if os.path.exists(kaggle_json) or env_configured:
        return True

    print("=" * 70)
    print("KAGGLE API CREDENTIALS NOT FOUND")
    print("=" * 70)
    print()
    print("To download datasets automatically, you need Kaggle API credentials.")
    print()
    print("Option 1: API Token File (Recommended)")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Scroll to 'API' section → Click 'Create New Token'")
    print("  3. This downloads kaggle.json")
    print(f"  4. Move it to: {kaggle_json}")
    print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
    print()
    print("Option 2: Environment Variables")
    print("  export KAGGLE_USERNAME=your_username")
    print("  export KAGGLE_KEY=your_api_key")
    print()
    print("Option 3: Manual Download")
    print("  Download each dataset from Kaggle website and extract to:")
    print(f"  {RAW_DATA_DIR}/<dataset_name>/")
    print()
    print("  Expected datasets:")
    for name, info in KAGGLE_DATASETS.items():
        print(f"    - {RAW_DATA_DIR}/{name}/ → from {info['kaggle_path']}")
    print()
    return False


def download_dataset(name, kaggle_path, dest_dir):
    """Download and extract a single Kaggle dataset."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest_path = os.path.join(dest_dir, name)

    # Skip if already downloaded
    if os.path.exists(dest_path) and any(Path(dest_path).rglob("*")):
        print(f"  [SKIP] {name} already exists at {dest_path}")
        return True

    os.makedirs(dest_path, exist_ok=True)

    try:
        print(f"  [DOWNLOAD] {kaggle_path} → {dest_path}")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(kaggle_path, path=dest_path, unzip=True)
        print(f"  [OK] {name} downloaded successfully")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {name}: {e}")
        return False


def verify_dataset(name, dest_dir):
    """Verify a downloaded dataset has content."""
    dest_path = os.path.join(dest_dir, name)
    if not os.path.exists(dest_path):
        return False

    # Count image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_count = 0
    dir_count = 0
    for root, dirs, files in os.walk(dest_path):
        dir_count += len(dirs)
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                image_count += 1

    if image_count > 0:
        print(f"  [VERIFIED] {name}: {image_count} images in {dir_count} directories")
        return True
    else:
        print(f"  [WARNING] {name}: No images found at {dest_path}")
        return False


def main():
    print("=" * 70)
    print("FOOD RECOGNITION - DATASET DOWNLOADER")
    print("=" * 70)
    print()

    # Check credentials
    has_credentials = check_kaggle_credentials()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Check which datasets are already present
    already_present = []
    need_download = []
    for name, info in KAGGLE_DATASETS.items():
        dest_path = os.path.join(RAW_DATA_DIR, name)
        if os.path.exists(dest_path) and any(Path(dest_path).rglob("*")):
            already_present.append(name)
        else:
            need_download.append(name)

    if already_present:
        print(f"Datasets already present ({len(already_present)}):")
        for name in already_present:
            print(f"  ✓ {name}")
        print()

    if not need_download:
        print("All datasets are already downloaded!")
        print()
        # Verify all
        print("Verifying datasets...")
        for name in KAGGLE_DATASETS:
            verify_dataset(name, RAW_DATA_DIR)
        return

    print(f"Datasets to download ({len(need_download)}):")
    for name in need_download:
        info = KAGGLE_DATASETS[name]
        print(f"  • {name}: {info['description']}")
    print()

    if not has_credentials:
        print("Cannot download without Kaggle credentials.")
        print("Please set up credentials and run again, or download manually.")
        sys.exit(1)

    # Download each dataset
    results = {}
    for name in need_download:
        info = KAGGLE_DATASETS[name]
        print(f"\n--- {name} ---")
        success = download_dataset(name, info["kaggle_path"], RAW_DATA_DIR)
        results[name] = success

    # Verify all datasets
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    all_ok = True
    for name in KAGGLE_DATASETS:
        if not verify_dataset(name, RAW_DATA_DIR):
            all_ok = False

    if all_ok:
        print("\n✓ All datasets ready! Run prepare_data.py next.")
    else:
        print("\n⚠ Some datasets are missing. Check errors above.")


if __name__ == "__main__":
    main()
