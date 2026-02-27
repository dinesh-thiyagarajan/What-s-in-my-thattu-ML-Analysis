#!/usr/bin/env python3
"""
Train a MobileNetV2-based food recognition model using transfer learning.

Two-phase training:
  Phase 1: Feature extraction - freeze base, train classification head
  Phase 2: Fine-tuning - unfreeze top layers of base, train with lower LR

Usage:
    python train.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    BATCH_SIZE,
    CLASS_MAPPING_PATH,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    IMG_SIZE,
    LOG_DIR,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
    MODEL_DIR,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE2_EPOCHS,
    PHASE2_LR,
    RANDOM_SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


def setup_gpu():
    """Configure GPU memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    else:
        print("No GPU found. Training on CPU (will be slow).")


def load_datasets():
    """Load train, val, and test datasets from directory structure."""
    print("Loading datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
        seed=RANDOM_SEED,
    )

    # Get class names from training dataset
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes")
    print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Val batches: {tf.data.experimental.cardinality(val_ds).numpy()}")

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


def build_data_augmentation():
    """Build data augmentation pipeline as Keras layers."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_model(num_classes):
    """
    Build MobileNetV2-based model with custom classification head.

    Architecture:
        Input (224x224x3)
        → Data Augmentation
        → MobileNetV2 Preprocessing
        → MobileNetV2 Base (frozen initially)
        → GlobalAveragePooling2D
        → Dropout
        → Dense (num_classes, softmax)
    """
    # Input
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Data augmentation (only active during training)
    x = build_data_augmentation()(inputs)

    # MobileNetV2 preprocessing (scales pixels to [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze for Phase 1

    x = base_model(x, training=False)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model


def train_phase1(model, train_ds, val_ds):
    """Phase 1: Train only the classification head (base model frozen)."""
    print("\n" + "=" * 70)
    print("PHASE 1: Feature Extraction (base model frozen)")
    print("=" * 70)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, "phase1"),
            histogram_freq=1,
        ),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks,
    )

    return history1


def train_phase2(model, base_model, train_ds, val_ds):
    """Phase 2: Fine-tune top layers of the base model."""
    print("\n" + "=" * 70)
    print("PHASE 2: Fine-tuning (top layers unfrozen)")
    print("=" * 70)

    # Unfreeze the top ~30% of MobileNetV2 layers
    base_model.trainable = True
    num_layers = len(base_model.layers)
    fine_tune_at = int(num_layers * 0.7)  # Freeze bottom 70%

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"Base model layers: {num_layers}")
    print(f"Fine-tuning from layer {fine_tune_at} ({trainable_count} trainable layers)")

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, "best_model.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, "phase2"),
            histogram_freq=1,
        ),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks,
    )

    return history2, best_model_path


def plot_training_history(history1, history2):
    """Plot training and validation metrics across both phases."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Combine histories
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    phase1_epochs = len(history1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, acc, "b-", label="Train Accuracy")
    ax1.plot(epochs, val_acc, "r-", label="Val Accuracy")
    ax1.axvline(x=phase1_epochs, color="g", linestyle="--", label="Fine-tuning starts")
    ax1.set_title("Training & Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(epochs, loss, "b-", label="Train Loss")
    ax2.plot(epochs, val_loss, "r-", label="Val Loss")
    ax2.axvline(x=phase1_epochs, color="g", linestyle="--", label="Fine-tuning starts")
    ax2.set_title("Training & Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to {plot_path}")


def main():
    print("=" * 70)
    print("FOOD RECOGNITION - MODEL TRAINING")
    print("=" * 70)

    # Setup
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    setup_gpu()

    # Check data exists
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training data not found at {TRAIN_DIR}")
        print("Run prepare_data.py first.")
        return

    # Load class mapping
    if os.path.exists(CLASS_MAPPING_PATH):
        with open(CLASS_MAPPING_PATH) as f:
            class_mapping = json.load(f)
        print(f"Loaded class mapping: {len(class_mapping)} classes")

    # Load datasets
    train_ds, val_ds, class_names, num_classes = load_datasets()

    # Build model
    print("\nBuilding MobileNetV2 model...")
    model, base_model = build_model(num_classes)

    # Phase 1: Feature extraction
    history1 = train_phase1(model, train_ds, val_ds)

    # Phase 2: Fine-tuning
    history2, best_model_path = train_phase2(model, base_model, train_ds, val_ds)

    # Plot history
    plot_training_history(history1, history2)

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    print(f"Best model saved to {best_model_path}")

    # Print final metrics
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Phase 1 best val_accuracy: {max(history1.history['val_accuracy']):.4f}")
    print(f"  Phase 2 best val_accuracy: {max(history2.history['val_accuracy']):.4f}")
    print(f"\nNext step: python evaluate.py")


if __name__ == "__main__":
    main()
