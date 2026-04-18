"""Main entry point to train and test LSTM-based log failure prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing import parse_feature_sequence
from predict import predict_failure
from train_model import train_pipeline


def run() -> None:
    dataset_path = Path("dataset") / "Event_traces.csv"

    print("Starting training...")
    results = train_pipeline(
        dataset_path=dataset_path,
        sequence_length=5,
        step_size=1,
        max_records=10000,
        balanced_subset=True,
        epochs=4,
        target_precision=0.75,
    )

    print("\nTraining Results")
    print(f"Dataset: {results['dataset_path']}")
    print(f"Original Trace Count: {results['original_trace_count']}")
    print(f"Selected Trace Count: {results['selected_trace_count']}")
    print(f"Trace Count: {results['trace_count']}")
    print(f"Generated Windows: {results['window_count']}")
    print(f"Class Distribution (All): {results['class_distribution_all']}")
    print(f"Class Distribution (Train Before Balance): {results['class_distribution_train_before_balance']}")
    print(f"Class Distribution (Train After Balance): {results['class_distribution_train_after_balance']}")
    print(f"Class Distribution (Test): {results['class_distribution_test']}")
    print(f"Validation Samples: {results['samples_val']}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Target Precision: {results['target_precision']}")
    print(f"Decision Threshold: {results['decision_threshold']:.2f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    print(f"Training Plot Saved: {results['plot_path']}")
    print(f"Confusion Matrix Plot Saved: {results['confusion_matrix_plot_path']}")
    print(f"Metrics Plot Saved: {results['metrics_plot_path']}")

    # Use the first trace from dataset for a valid prediction demo.
    first_features = pd.read_csv(dataset_path).loc[0, "Features"]
    sample_sequence = parse_feature_sequence(first_features)[:5]
    try:
        prediction = predict_failure(sample_sequence)

        print("\nSample Prediction")
        print(f"Input Sequence: {prediction['input_sequence']}")
        print(f"Anomaly Probability: {prediction['anomaly_probability']:.4f}")
        print(f"Decision Threshold: {prediction['decision_threshold']:.2f}")
        print(f"Potential Failure/Anomaly: {prediction['predicted_failure']}")
        if prediction["unknown_event_ids"]:
            print(f"Unknown EventIds: {prediction['unknown_event_ids']}")
    except Exception as error:
        print("\nSample Prediction")
        print(f"Skipped due to artifact load error: {error}")


if __name__ == "__main__":
    run()
