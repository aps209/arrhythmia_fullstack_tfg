"""
config.py - Configuracion centralizada del backend ECG.
"""

from __future__ import annotations

import os
from pathlib import Path

from .wfdb_tools import CLASS_NAMES


ARTIFACTS_ENV = os.getenv("ARTIFACTS_PATH", os.getenv("MODEL_DIR", ""))
if ARTIFACTS_ENV:
    ARTIFACTS_DIR = Path(ARTIFACTS_ENV)
else:
    ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

DEFAULT_TARGET_FS = int(os.getenv("TARGET_FS", "250"))
DEFAULT_WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "256"))
DEFAULT_PRE_SAMPLES = int(os.getenv("PRE_SAMPLES", "96"))

MODEL_SPECS = {
    "conv1d_bilstm": {
        "label": "Conv1D + BiLSTM",
        "architecture_name": "Conv1D_BiLSTM_ECG",
        "subdir": "conv1d_bilstm",
        "model_version": "ecg-conv1d-bilstm-1.0.0",
        "legacy_root": True,
    },
    "lstm": {
        "label": "LSTM",
        "architecture_name": "LSTM_ECG",
        "subdir": "lstm",
        "model_version": "ecg-lstm-1.0.0",
        "legacy_root": False,
    },
    "simple_rnn": {
        "label": "Simple RNN",
        "architecture_name": "SimpleRNN_ECG",
        "subdir": "simple_rnn",
        "model_version": "ecg-simple-rnn-1.0.0",
        "legacy_root": False,
    },
}
DEFAULT_MODEL_KEY = "conv1d_bilstm"


def model_dir_for(model_key: str) -> Path:
    spec = MODEL_SPECS[model_key]
    preferred_dir = ARTIFACTS_DIR / spec["subdir"]
    preferred_model = preferred_dir / "model.keras"
    preferred_metadata = preferred_dir / "metadata.json"
    if preferred_model.exists() or preferred_metadata.exists():
        return preferred_dir

    legacy_model = ARTIFACTS_DIR / "model.keras"
    legacy_metadata = ARTIFACTS_DIR / "metadata.json"
    if spec.get("legacy_root") and (legacy_model.exists() or legacy_metadata.exists()):
        return ARTIFACTS_DIR
    return preferred_dir


def model_paths_for(model_key: str) -> tuple[Path, Path]:
    model_dir = model_dir_for(model_key)
    return model_dir / "model.keras", model_dir / "metadata.json"


def default_metadata_for(model_key: str) -> dict:
    spec = MODEL_SPECS[model_key]
    return {
        "model_key": model_key,
        "model_label": spec["label"],
        "model_version": spec["model_version"],
        "architecture_name": spec["architecture_name"],
        "input_type": "wfdb_ecg_signal",
        "target_fs": DEFAULT_TARGET_FS,
        "window_size": DEFAULT_WINDOW_SIZE,
        "pre_samples": DEFAULT_PRE_SAMPLES,
        "class_names": CLASS_NAMES,
    }
