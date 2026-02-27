#!/usr/bin/env python3
"""
Evaluate the trained food recognition model on the test set.

Generates:
  - Top-1 and Top-5 accuracy
  - Per-class precision, recall, F1 scores
  - Confusion matrix visualization (top confused classes)
  - Evaluation report text file

Usage:
    python evaluate.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    BATCH_SIZE,
    CLASS_MAPPING_PATH,
    IMG_SIZE,
    LABELS_PATH,
    MODEL_DIR,
    RANDOM_SEED,
    TEST_DIR,
)


def load_model():
    """Load the best trained model."""
    best_model_path = os.path.join(MODEL_DIR, "best_model.keras")
    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")

    model_path = best_model_path if os.path.exists(best_model_path) else final_model_path

    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        print("Run train.py first.")
        return None, None

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model, model_path


def load_test_data():
    """Load test dataset."""
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test data not found at {TEST_DIR}")
        return None, None

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        seed=RANDOM_SEED,
    )

    class_names = test_ds.class_names
    print(f"Test set: {len(class_names)} classes")

    return test_ds, class_names


def compute_predictions(model, test_ds):
    """Run inference on the entire test set."""
    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        all_probabilities.extend(predictions)
        all_predictions.extend(np.argmax(predictions, axis=1))
        all_labels.extend(labels.numpy())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def compute_topk_accuracy(labels, probabilities, k=5):
    """Compute top-k accuracy."""
    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
    correct = sum(1 for label, top_k in zip(labels, top_k_preds) if label in top_k)
    return correct / len(labels)


def plot_confusion_matrix(labels, predictions, class_names, top_n=25):
    """Plot confusion matrix for the top-N most confused classes."""
    cm = confusion_matrix(labels, predictions)

    # Find classes with most errors
    errors_per_class = []
    for i in range(len(cm)):
        total_errors = sum(cm[i]) - cm[i][i]
        errors_per_class.append((i, total_errors))

    errors_per_class.sort(key=lambda x: x[1], reverse=True)
    top_confused_indices = [idx for idx, _ in errors_per_class[:top_n]]
    top_confused_indices.sort()

    # Extract sub-matrix
    sub_cm = cm[np.ix_(top_confused_indices, top_confused_indices)]
    sub_names = [class_names[i][:20] for i in top_confused_indices]  # Truncate names

    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        sub_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sub_names,
        yticklabels=sub_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (Top {top_n} Most Confused Classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")


def save_report(report_text, top1_acc, top5_acc, model_path):
    """Save evaluation report to file."""
    report_path = os.path.join(MODEL_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FOOD RECOGNITION MODEL - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test directory: {TEST_DIR}\n\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)\n\n")
        f.write("Per-Class Classification Report:\n")
        f.write("-" * 70 + "\n")
        f.write(report_text)
    print(f"Evaluation report saved to {report_path}")


def main():
    print("=" * 70)
    print("FOOD RECOGNITION - MODEL EVALUATION")
    print("=" * 70)
    print()

    # Load model
    model, model_path = load_model()
    if model is None:
        return

    # Load test data
    test_ds, class_names = load_test_data()
    if test_ds is None:
        return

    # Compute predictions
    print("\nRunning inference on test set...")
    labels, predictions, probabilities = compute_predictions(model, test_ds)
    print(f"Total test samples: {len(labels)}")

    # Top-1 accuracy
    top1_acc = np.mean(labels == predictions)
    print(f"\nTop-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")

    # Top-5 accuracy
    top5_acc = compute_topk_accuracy(labels, probabilities, k=5)
    print(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")

    # Classification report
    print("\nGenerating classification report...")
    report = classification_report(
        labels,
        predictions,
        target_names=[n.replace("_", " ") for n in class_names],
        zero_division=0,
    )
    print(report)

    # Confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(labels, predictions, class_names)

    # Save report
    save_report(report, top1_acc, top5_acc, model_path)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc*100:.2f}%")
    print("=" * 70)
    print("\nNext step: python convert_tflite.py")


if __name__ == "__main__":
    main()
