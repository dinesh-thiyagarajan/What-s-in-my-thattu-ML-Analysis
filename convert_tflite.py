#!/usr/bin/env python3
"""
Convert the trained Keras model to TensorFlow Lite format for Android deployment.

Generates two TFLite models:
  1. Float16 quantized (recommended) - ~50% size reduction, minimal accuracy loss
  2. Int8 quantized (optional) - ~75% size reduction, for CPU-only devices

Also generates labels.txt for Android integration.

Usage:
    python convert_tflite.py
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

from config import (
    CLASS_MAPPING_PATH,
    IMG_SIZE,
    LABELS_PATH,
    MODEL_DIR,
    NUM_CALIBRATION_IMAGES,
    RANDOM_SEED,
    TFLITE_FLOAT16_PATH,
    TFLITE_INT8_PATH,
    TRAIN_DIR,
)


def load_trained_model():
    """Load the best trained Keras model."""
    best_model_path = os.path.join(MODEL_DIR, "best_model.keras")
    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")

    model_path = best_model_path if os.path.exists(best_model_path) else final_model_path

    if not os.path.exists(model_path):
        print(f"Error: No model found. Checked:")
        print(f"  {best_model_path}")
        print(f"  {final_model_path}")
        print("Run train.py first.")
        sys.exit(1)

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def convert_float16(model):
    """
    Convert to TFLite with float16 quantization.

    Benefits:
    - ~50% size reduction
    - Negligible accuracy loss
    - Can use GPU delegate on Android for fast inference
    """
    print("\n--- Float16 Quantization ---")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(TFLITE_FLOAT16_PATH), exist_ok=True)
    with open(TFLITE_FLOAT16_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Float16 model saved to: {TFLITE_FLOAT16_PATH}")
    print(f"Model size: {size_mb:.2f} MB")

    return tflite_model


def get_representative_dataset():
    """
    Generator that yields representative samples for int8 calibration.
    Uses a subset of training images.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        label_mode=None,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    count = 0
    for image_batch in dataset:
        if count >= NUM_CALIBRATION_IMAGES:
            break
        # Scale to [0, 1] range as the model expects
        yield [image_batch.numpy().astype(np.float32)]
        count += 1


def convert_int8(model):
    """
    Convert to TFLite with full int8 quantization.

    Benefits:
    - ~75% size reduction
    - Fast inference on CPU (uses NEON on ARM)
    - Requires representative dataset for calibration
    """
    print("\n--- Int8 Quantization ---")

    if not os.path.exists(TRAIN_DIR):
        print("Skipping int8 quantization: training data not found for calibration")
        return None

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = get_representative_dataset
    # Keep input/output as float for easier integration
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Int8 quantization failed: {e}")
        print("This is non-critical. Use the float16 model instead.")
        return None

    os.makedirs(os.path.dirname(TFLITE_INT8_PATH), exist_ok=True)
    with open(TFLITE_INT8_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Int8 model saved to: {TFLITE_INT8_PATH}")
    print(f"Model size: {size_mb:.2f} MB")

    return tflite_model


def generate_labels():
    """Generate labels.txt for Android from class mapping."""
    if os.path.exists(LABELS_PATH):
        print(f"\nLabels file already exists at {LABELS_PATH}")
        with open(LABELS_PATH) as f:
            lines = f.readlines()
        print(f"  Contains {len(lines)} classes")
        return

    if os.path.exists(CLASS_MAPPING_PATH):
        with open(CLASS_MAPPING_PATH) as f:
            class_mapping = json.load(f)

        id_to_class = {v: k for k, v in class_mapping.items()}
        with open(LABELS_PATH, "w") as f:
            for i in range(len(id_to_class)):
                f.write(id_to_class[i] + "\n")
        print(f"\nLabels file generated: {LABELS_PATH} ({len(id_to_class)} classes)")
    else:
        print(f"\nWarning: class_mapping.json not found at {CLASS_MAPPING_PATH}")
        print("Labels file not generated. Run prepare_data.py to create it.")


def verify_tflite_model(tflite_path):
    """Verify a TFLite model can load and run inference."""
    if not os.path.exists(tflite_path):
        return False

    print(f"\nVerifying {os.path.basename(tflite_path)}...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input shape:  {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    # Test with random input
    input_shape = input_details[0]["shape"]
    test_input = np.random.rand(*input_shape).astype(np.float32) * 255.0
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    num_classes = output.shape[-1]
    print(f"  Output classes: {num_classes}")
    print(f"  Output sum: {output.sum():.4f} (should be ~1.0 for softmax)")
    print(f"  Verification: PASSED")

    return True


def main():
    print("=" * 70)
    print("FOOD RECOGNITION - TFLITE CONVERSION")
    print("=" * 70)

    # Load model
    model = load_trained_model()

    # Get original model size from the .keras file
    best_model_path = os.path.join(MODEL_DIR, "best_model.keras")
    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
    source_path = best_model_path if os.path.exists(best_model_path) else final_model_path
    original_size_mb = os.path.getsize(source_path) / (1024 * 1024)
    print(f"Original model size: {original_size_mb:.2f} MB")

    # Convert to float16
    float16_model = convert_float16(model)

    # Convert to int8
    int8_model = convert_int8(model)

    # Generate labels
    generate_labels()

    # Verify models
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    verify_tflite_model(TFLITE_FLOAT16_PATH)
    if int8_model:
        verify_tflite_model(TFLITE_INT8_PATH)

    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nFiles for Android deployment:")
    print(f"  Model (float16): {TFLITE_FLOAT16_PATH}")
    if int8_model:
        print(f"  Model (int8):    {TFLITE_INT8_PATH}")
    print(f"  Labels:          {LABELS_PATH}")
    print(f"\nCopy these files to your Android app's assets/ folder.")
    print(f"\nNext step: python test_inference.py")


if __name__ == "__main__":
    main()
