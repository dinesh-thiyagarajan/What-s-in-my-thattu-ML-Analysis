#!/usr/bin/env python3
"""
Test the TFLite model with sample images to verify it works correctly.

This simulates how the model will be used on Android:
  1. Load TFLite model
  2. Load and preprocess an image
  3. Run inference
  4. Get top-K predictions with confidence scores

Usage:
    python test_inference.py                          # Test with random test images
    python test_inference.py --image path/to/food.jpg # Test with a specific image
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

from config import (
    IMG_SIZE,
    LABELS_PATH,
    MODEL_DIR,
    TEST_DIR,
    TFLITE_FLOAT16_PATH,
    TFLITE_INT8_PATH,
)


def load_labels():
    """Load class labels from labels.txt."""
    if not os.path.exists(LABELS_PATH):
        print(f"Error: Labels file not found at {LABELS_PATH}")
        print("Run prepare_data.py or convert_tflite.py first.")
        sys.exit(1)

    with open(LABELS_PATH) as f:
        labels = [line.strip() for line in f.readlines()]

    print(f"Loaded {len(labels)} class labels")
    return labels


def load_tflite_model(model_path):
    """Load a TFLite model and allocate tensors."""
    # Import TFLite runtime (works without full TensorFlow too)
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Model loaded: {os.path.basename(model_path)}")
    print(f"  Input:  {input_details[0]['shape']} ({input_details[0]['dtype'].__name__})")
    print(f"  Output: {output_details[0]['shape']} ({output_details[0]['dtype'].__name__})")

    return interpreter, input_details, output_details


def preprocess_image(image_path):
    """
    Load and preprocess an image for MobileNetV2 inference.

    Steps:
    1. Open image and convert to RGB
    2. Resize to 224x224
    3. Convert to float32 numpy array
    4. Keep pixel values in [0, 255] range (preprocessing is baked into the model)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_inference(interpreter, input_details, output_details, image_array):
    """Run inference and return prediction probabilities."""
    interpreter.set_tensor(input_details[0]["index"], image_array)

    start_time = time.perf_counter()
    interpreter.invoke()
    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0], inference_time


def get_top_predictions(probabilities, labels, top_k=5):
    """Get top-K predictions with class names and confidence scores."""
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "class_id": int(idx),
            "class_name": labels[idx] if idx < len(labels) else f"unknown_{idx}",
            "confidence": float(probabilities[idx]),
        })
    return results


def find_sample_images(n=5):
    """Find random sample images from the test set."""
    if not os.path.exists(TEST_DIR):
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = []

    for root, dirs, files in os.walk(TEST_DIR):
        for f in files:
            if os.path.splitext(f)[1].lower() in image_extensions:
                all_images.append(os.path.join(root, f))

    if not all_images:
        return []

    np.random.seed(42)
    indices = np.random.choice(len(all_images), min(n, len(all_images)), replace=False)
    return [all_images[i] for i in indices]


def test_single_image(interpreter, input_details, output_details, labels, image_path):
    """Test inference on a single image and display results."""
    # Get true class from directory name
    parent_dir = os.path.basename(os.path.dirname(image_path))
    true_class = parent_dir.replace("_", " ")

    # Preprocess and run inference
    img_array = preprocess_image(image_path)
    probabilities, inference_time = run_inference(
        interpreter, input_details, output_details, img_array
    )

    # Get top predictions
    top_preds = get_top_predictions(probabilities, labels, top_k=5)

    # Display results
    print(f"\n  Image: {os.path.basename(image_path)}")
    print(f"  True class: {true_class}")
    print(f"  Inference time: {inference_time:.1f} ms")
    print(f"  Top-5 predictions:")
    for i, pred in enumerate(top_preds):
        marker = " ✓" if pred["class_name"] == true_class else ""
        print(f"    {i+1}. {pred['class_name']}: {pred['confidence']*100:.2f}%{marker}")

    return top_preds[0]["class_name"] == true_class, inference_time


def main():
    parser = argparse.ArgumentParser(description="Test TFLite food recognition model")
    parser.add_argument("--image", type=str, help="Path to a specific image to test")
    parser.add_argument("--model", type=str, default="float16",
                       choices=["float16", "int8"],
                       help="TFLite model variant to test")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of random test images to use")
    args = parser.parse_args()

    print("=" * 70)
    print("FOOD RECOGNITION - TFLITE INFERENCE TEST")
    print("=" * 70)

    # Select model
    model_path = TFLITE_FLOAT16_PATH if args.model == "float16" else TFLITE_INT8_PATH
    if not os.path.exists(model_path):
        print(f"Error: TFLite model not found at {model_path}")
        print("Run convert_tflite.py first.")
        sys.exit(1)

    # Load model and labels
    labels = load_labels()
    interpreter, input_details, output_details = load_tflite_model(model_path)

    # Determine images to test
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = find_sample_images(n=args.num_samples)
        if not image_paths:
            print(f"No test images found in {TEST_DIR}")
            print("Use --image flag to test with a specific image.")
            sys.exit(1)

    # Run tests
    print(f"\nTesting with {len(image_paths)} images...")
    print("-" * 70)

    correct = 0
    total_time = 0

    for img_path in image_paths:
        try:
            is_correct, inf_time = test_single_image(
                interpreter, input_details, output_details, labels, img_path
            )
            if is_correct:
                correct += 1
            total_time += inf_time
        except Exception as e:
            print(f"\n  Error processing {img_path}: {e}")

    # Summary
    n = len(image_paths)
    avg_time = total_time / n if n > 0 else 0

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Images tested: {n}")
    print(f"  Correct (top-1): {correct}/{n} ({correct/n*100:.1f}%)" if n > 0 else "")
    print(f"  Avg inference time: {avg_time:.1f} ms")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Model size: {os.path.getsize(model_path)/(1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
