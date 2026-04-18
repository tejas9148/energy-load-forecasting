"""Data loading and preprocessing for HDFS trace-level failure prediction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


REQUIRED_COLUMNS = ["Label", "Features"]
EVENT_PATTERN = re.compile(r"E\d+")


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load Event_traces.csv file and validate required columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    return df


def parse_feature_sequence(feature_text: str) -> List[str]:
    """Extract EventId tokens (E1, E2, ...) from a trace Features string."""
    if not isinstance(feature_text, str):
        return []
    return EVENT_PATTERN.findall(feature_text)


def encode_event_ids(df: pd.DataFrame) -> Tuple[List[np.ndarray], np.ndarray, LabelEncoder]:
    """Encode trace event sequences and map labels into binary classes."""
    processed_df = df.copy()
    processed_df["Label"] = processed_df["Label"].astype(str).str.strip().str.lower()

    label_map = {
        "success": 0,
        "normal": 0,
        "fail": 1,
        "anomaly": 1,
    }
    if not processed_df["Label"].isin(label_map).all():
        unknown = processed_df.loc[~processed_df["Label"].isin(label_map), "Label"].unique().tolist()
        raise ValueError(f"Unsupported labels found in dataset: {unknown}")

    trace_labels = processed_df["Label"].map(label_map).to_numpy(dtype=np.float32)

    trace_tokens = processed_df["Features"].apply(parse_feature_sequence)
    trace_tokens = trace_tokens[trace_tokens.apply(len) > 0]
    valid_indices = trace_tokens.index.to_numpy()
    trace_labels = trace_labels[valid_indices]

    all_tokens = [token for tokens in trace_tokens for token in tokens]
    if not all_tokens:
        raise ValueError("No EventId tokens found inside Features column.")

    encoder = LabelEncoder()
    encoder.fit(all_tokens)

    encoded_traces: List[np.ndarray] = []
    for tokens in trace_tokens:
        encoded = encoder.transform(tokens).astype(np.int32)
        encoded_traces.append(encoded)

    return encoded_traces, trace_labels, encoder
