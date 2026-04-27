from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.wfdb_tools import (  # noqa: E402
    AAMI_MAPPING,
    CLASS_NAMES,
    load_annotations,
    load_record,
    segment_windows,
)


def build_dataset(
    data_dir: str,
    target_fs: int = 250,
    window_size: int = 256,
    pre_samples: int = 96,
    preferred_lead: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye dataset supervisado de latidos ECG reales centrados en anotaciones AAMI."""
    record_files = sorted(glob.glob(str(Path(data_dir) / "*.hea")))
    record_ids = [Path(filepath).stem for filepath in record_files]

    X, y, groups = [], [], []

    for record_id in record_ids:
        record_path = Path(data_dir) / record_id
        try:
            loaded = load_record(record_path, target_fs=target_fs, preferred_lead=preferred_lead)
            annotations = load_annotations(record_path, original_fs=loaded.original_fs, target_fs=loaded.target_fs)
            if annotations is None:
                continue

            segments, valid_centers = segment_windows(
                loaded.filtered_signal,
                annotations.samples,
                window_size=window_size,
                pre_samples=pre_samples,
            )
            if len(segments) == 0:
                continue

            centers_set = set(valid_centers.tolist())
            label_map = {
                int(sample): AAMI_MAPPING[symbol]
                for sample, symbol in zip(annotations.samples, annotations.symbols)
                if int(sample) in centers_set and symbol in AAMI_MAPPING
            }

            record_labels = []
            record_segments = []
            for segment, center in zip(segments, valid_centers):
                label = label_map.get(int(center))
                if label is None:
                    continue
                record_segments.append(segment[:, np.newaxis])
                record_labels.append(label)

            if not record_segments:
                continue

            X.extend(record_segments)
            y.extend(record_labels)
            groups.extend([record_id] * len(record_labels))
        except Exception:
            continue

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups),
    )
