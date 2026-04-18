"""Training pipeline for trace-level HDFS failure prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from lstm_model import build_lstm_model
from preprocessing import encode_event_ids, load_dataset
from sequence_generator import generate_sequences


ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODEL_PATH = ARTIFACTS_DIR / "lstm_failure_predictor.keras"
ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.joblib"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"
TRAINING_PLOT_PATH = PLOTS_DIR / "training_history.png"
CONFUSION_MATRIX_PLOT_PATH = PLOTS_DIR / "confusion_matrix.png"
METRICS_PLOT_PATH = PLOTS_DIR / "metrics_bar.png"


def _sample_balanced_subset(
    df,
    max_records: int,
    random_state: int,
):
    """Create an exact balanced subset with half normal/success and half fail/anomaly."""
    if max_records <= 0 or max_records % 2 != 0:
        raise ValueError("max_records must be a positive even number.")

    labels = df["Label"].astype(str).str.strip().str.lower()
    normal_idx = labels[labels.isin(["success", "normal"])].index
    anomaly_idx = labels[labels.isin(["fail", "anomaly"])].index

    per_class = max_records // 2
    if len(normal_idx) < per_class or len(anomaly_idx) < per_class:
        raise ValueError(
            f"Not enough records for balanced subset of {max_records}. "
            f"Available normal/success={len(normal_idx)}, fail/anomaly={len(anomaly_idx)}"
        )

    rng = np.random.default_rng(seed=random_state)
    sampled_normal = rng.choice(normal_idx.to_numpy(), size=per_class, replace=False)
    sampled_anomaly = rng.choice(anomaly_idx.to_numpy(), size=per_class, replace=False)
    selected_idx = np.concatenate([sampled_normal, sampled_anomaly])
    rng.shuffle(selected_idx)
    return df.loc[selected_idx].reset_index(drop=True)


def _save_artifacts(
    sequence_length: int,
    step_size: int,
    decision_threshold: float,
    label_encoder,
) -> None:
    """Persist model metadata and encoder for later inference."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(label_encoder, ENCODER_PATH)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_type": "HDFS_Event_traces",
                "sequence_length": sequence_length,
                "step_size": step_size,
                "decision_threshold": decision_threshold,
                "model_path": str(MODEL_PATH),
                "encoder_path": str(ENCODER_PATH),
            },
            file,
            indent=2,
        )


def _plot_training_history(history) -> None:
    """Plot and save training accuracy and loss curves."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history.history:
        axes[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Training Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(TRAINING_PLOT_PATH, dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    """Plot and save confusion matrix heatmap using matplotlib only."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Anomaly (1)"])
    ax.set_yticklabels(["Normal (0)", "Anomaly (1)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=150)
    plt.close(fig)


def _plot_metrics(accuracy: float, precision: float, recall: float, f1: float) -> None:
    """Plot and save key classification metrics as a bar chart."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_values = [accuracy, precision, recall, f1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(metric_names, metric_values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_title("Evaluation Metrics")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")

    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")

    fig.tight_layout()
    fig.savefig(METRICS_PLOT_PATH, dpi=150)
    plt.close(fig)


def _balance_training_set(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance training classes by random oversampling the minority class."""
    y_int = y_train.astype(int)
    class_counts = np.bincount(y_int, minlength=2)
    if class_counts[0] == 0 or class_counts[1] == 0:
        return x_train, y_train

    majority_class = int(np.argmax(class_counts))
    minority_class = 1 - majority_class
    majority_count = int(class_counts[majority_class])
    minority_count = int(class_counts[minority_class])

    if minority_count >= majority_count:
        return x_train, y_train

    rng = np.random.default_rng(seed=random_state)
    minority_indices = np.where(y_int == minority_class)[0]
    oversampled_indices = rng.choice(
        minority_indices,
        size=majority_count - minority_count,
        replace=True,
    )

    x_balanced = np.concatenate([x_train, x_train[oversampled_indices]], axis=0)
    y_balanced = np.concatenate([y_train, y_train[oversampled_indices]], axis=0)

    shuffle_idx = rng.permutation(len(y_balanced))
    return x_balanced[shuffle_idx], y_balanced[shuffle_idx]


def _ensure_test_has_anomaly(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Swap one sample so test contains at least one anomaly if available."""
    if np.sum(y_test == 1) > 0 or np.sum(y_train == 1) == 0:
        return x_train, y_train, x_test, y_test

    anomaly_idx_train = int(np.where(y_train == 1)[0][0])
    normal_idx_test = int(np.where(y_test == 0)[0][0])

    x_anomaly = x_train[anomaly_idx_train].copy()
    y_anomaly = y_train[anomaly_idx_train].copy()
    x_normal = x_test[normal_idx_test].copy()
    y_normal = y_test[normal_idx_test].copy()

    x_train[anomaly_idx_train] = x_normal
    y_train[anomaly_idx_train] = y_normal
    x_test[normal_idx_test] = x_anomaly
    y_test[normal_idx_test] = y_anomaly
    return x_train, y_train, x_test, y_test


def _find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float | None = None,
) -> float:
    """Choose threshold from validation curve based on precision target or best F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true.astype(int), y_prob)
    if len(thresholds) == 0:
        return 0.5

    if target_precision is not None:
        valid_indices = np.where(precisions[:-1] >= target_precision)[0]
        if len(valid_indices) > 0:
            # Among thresholds meeting precision target, maximize recall.
            best_idx = int(valid_indices[np.argmax(recalls[:-1][valid_indices])])
            return float(thresholds[best_idx])

    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def train_pipeline(
    dataset_path: str | Path,
    sequence_length: int = 5,
    step_size: int = 1,
    max_records: int | None = None,
    balanced_subset: bool = False,
    test_size: float = 0.20,
    random_state: int = 42,
    epochs: int = 5,
    batch_size: int = 64,
    decision_threshold: float = 0.5,
    tune_threshold: bool = True,
    target_precision: float | None = None,
) -> Dict[str, object]:
    """Run full preprocessing, sequence creation, training, and evaluation."""
    df = load_dataset(dataset_path)
    original_trace_count = len(df)

    if balanced_subset and max_records is not None:
        df = _sample_balanced_subset(df=df, max_records=max_records, random_state=random_state)
    elif max_records is not None:
        df = df.sample(n=min(max_records, len(df)), random_state=random_state).reset_index(drop=True)

    encoded_traces, trace_labels, label_encoder = encode_event_ids(df)

    x, y, sequence_meta = generate_sequences(
        encoded_traces=encoded_traces,
        trace_labels=trace_labels,
        sequence_length=sequence_length,
        step_size=step_size,
    )

    all_class_distribution = {
        "normal_0": int(np.sum(y == 0)),
        "anomaly_1": int(np.sum(y == 1)),
    }

    unique_classes = np.unique(y)
    stratify_target = y if len(unique_classes) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    x_train, y_train, x_test, y_test = _ensure_test_has_anomaly(x_train, y_train, x_test, y_test)

    # Create a validation split for better early stopping and threshold tuning.
    train_unique_classes = np.unique(y_train)
    stratify_train = y_train if len(train_unique_classes) > 1 else None
    x_train_main, x_val, y_train_main, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=random_state,
        stratify=stratify_train,
    )

    train_distribution_before = {
        "normal_0": int(np.sum(y_train_main == 0)),
        "anomaly_1": int(np.sum(y_train_main == 1)),
    }

    x_train_balanced, y_train_balanced = _balance_training_set(
        x_train=x_train_main,
        y_train=y_train_main,
        random_state=random_state,
    )

    train_distribution_after = {
        "normal_0": int(np.sum(y_train_balanced == 0)),
        "anomaly_1": int(np.sum(y_train_balanced == 1)),
    }
    test_distribution = {
        "normal_0": int(np.sum(y_test == 0)),
        "anomaly_1": int(np.sum(y_test == 1)),
    }

    model = build_lstm_model(
        vocab_size=len(label_encoder.classes_),
        sequence_length=sequence_length,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    history = model.fit(
        x_train_balanced,
        y_train_balanced,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    if tune_threshold:
        val_prob = model.predict(x_val, verbose=0).flatten()
        decision_threshold = _find_best_threshold(
            y_val,
            val_prob,
            target_precision=target_precision,
        )

    y_prob = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_prob >= decision_threshold).astype(int)
    cm = confusion_matrix(y_test.astype(int), y_pred, labels=[0, 1])
    precision = precision_score(y_test.astype(int), y_pred, zero_division=0)
    recall = recall_score(y_test.astype(int), y_pred, zero_division=0)
    f1 = f1_score(y_test.astype(int), y_pred, zero_division=0)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    _save_artifacts(
        sequence_length=sequence_length,
        step_size=step_size,
        decision_threshold=decision_threshold,
        label_encoder=label_encoder,
    )
    _plot_training_history(history)
    _plot_confusion_matrix(cm)
    _plot_metrics(float(test_accuracy), float(precision), float(recall), float(f1))

    return {
        "dataset_path": str(dataset_path),
        "original_trace_count": int(original_trace_count),
        "selected_trace_count": int(len(df)),
        "trace_count": int(len(encoded_traces)),
        "window_count": int(sequence_meta["samples"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "decision_threshold": float(decision_threshold),
        "target_precision": None if target_precision is None else float(target_precision),
        "confusion_matrix": cm.tolist(),
        "class_distribution_all": all_class_distribution,
        "class_distribution_train_before_balance": train_distribution_before,
        "class_distribution_train_after_balance": train_distribution_after,
        "class_distribution_test": test_distribution,
        "model_path": str(MODEL_PATH),
        "encoder_path": str(ENCODER_PATH),
        "plot_path": str(TRAINING_PLOT_PATH),
        "confusion_matrix_plot_path": str(CONFUSION_MATRIX_PLOT_PATH),
        "metrics_plot_path": str(METRICS_PLOT_PATH),
        "samples_train": int(len(x_train_balanced)),
        "samples_val": int(len(x_val)),
        "samples_test": int(len(x_test)),
    }


if __name__ == "__main__":
    results = train_pipeline(dataset_path=Path("dataset") / "Event_traces.csv")
    print("Training complete")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
