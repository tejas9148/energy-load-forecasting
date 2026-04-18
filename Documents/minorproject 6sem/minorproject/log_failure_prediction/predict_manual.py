"""Interactive CLI for manual sequence prediction."""

from __future__ import annotations

from predict import predict_failure


def main() -> None:
    print("Manual Failure Prediction")
    print("Enter EventIds separated by commas (example: E5,E22,E5,E5,E11)")
    raw = input("Sequence: ").strip()

    if not raw:
        print("No input provided.")
        return

    sequence = [token.strip() for token in raw.split(",") if token.strip()]

    try:
        result = predict_failure(sequence)
    except Exception as error:
        print(f"Prediction failed: {error}")
        return

    print("\nPrediction Result")
    print(f"Input Sequence: {result['input_sequence']}")
    print(f"Anomaly Probability: {result['anomaly_probability']:.4f}")
    print(f"Decision Threshold: {result['decision_threshold']:.4f}")
    print(f"Potential Failure/Anomaly: {result['predicted_failure']}")
    if result["unknown_event_ids"]:
        print(f"Unknown EventIds: {result['unknown_event_ids']}")


if __name__ == "__main__":
    main()
