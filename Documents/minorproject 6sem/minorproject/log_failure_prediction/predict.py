"""Inference utilities for sequence-level failure/anomaly prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from tensorflow.keras.models import load_model


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lstm_failure_predictor.keras"
ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.joblib"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"


def _load_config() -> Dict[str, object]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Model config not found. Train the model first using train_model.py or main.py."
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _encode_sequence(event_sequence: List[str], label_encoder, sequence_length: int) -> np.ndarray:
    """Convert string EventIds to encoded numeric sequence for model input."""
    if len(event_sequence) != sequence_length:
        raise ValueError(f"Input sequence must have length {sequence_length}")

    known_classes = set(label_encoder.classes_)
    encoded = []

    # Unknown EventIds are mapped to 0 to keep shape valid, but flagged in result.
    for event_id in event_sequence:
        if event_id in known_classes:
            encoded_value = int(label_encoder.transform([event_id])[0])
        else:
            encoded_value = 0
        encoded.append(encoded_value)

    return np.array(encoded, dtype=np.int32).reshape(1, sequence_length)


def predict_failure(event_sequence: List[str]) -> Dict[str, object]:
    """Predict failure/anomaly probability for a sequence of EventIds."""
    config = _load_config()

    sequence_length = int(config["sequence_length"])
    decision_threshold = float(config.get("decision_threshold", 0.5))
    model = load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    known_classes = set(label_encoder.classes_)
    unknown_events = [event for event in event_sequence if event not in known_classes]

    x_input = _encode_sequence(
        event_sequence=event_sequence,
        label_encoder=label_encoder,
        sequence_length=sequence_length,
    )

    probability = float(model.predict(x_input, verbose=0)[0][0])
    is_anomaly = bool(probability >= decision_threshold or len(unknown_events) > 0)

    return {
        "input_sequence": event_sequence,
        "anomaly_probability": probability,
        "decision_threshold": decision_threshold,
        "predicted_failure": is_anomaly,
        "unknown_event_ids": unknown_events,
    }
