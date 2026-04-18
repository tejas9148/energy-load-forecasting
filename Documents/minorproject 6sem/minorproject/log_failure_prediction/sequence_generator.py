"""Sequence generation utilities for trace-level LSTM training."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def generate_sequences(
    encoded_traces: list[np.ndarray],
    trace_labels: np.ndarray,
    sequence_length: int = 5,
    step_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Create fixed-length input windows from each trace.

    Each generated sequence inherits the parent trace label:
    0 = normal/success, 1 = anomaly/fail.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")
    if step_size <= 0:
        raise ValueError("step_size must be greater than 0")

    x_data = []
    y_data = []

    for trace, label in zip(encoded_traces, trace_labels):
        if len(trace) < sequence_length:
            continue

        for idx in range(0, len(trace) - sequence_length + 1, step_size):
            seq = trace[idx : idx + sequence_length]
            x_data.append(seq)
            y_data.append(int(label))

    if not x_data:
        raise ValueError("No sequences generated. Check sequence_length and dataset contents.")

    x = np.array(x_data, dtype=np.int32)
    y = np.array(y_data, dtype=np.float32)

    metadata = {
        "sequence_length": float(sequence_length),
        "samples": float(len(x)),
        "normal_samples": float(np.sum(y == 0)),
        "anomaly_samples": float(np.sum(y == 1)),
    }
    return x, y, metadata
