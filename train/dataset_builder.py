from pathlib import Path
from typing import Tuple
import glob

import numpy as np
import wfdb

AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4,
}

VALID_BEATS = set(AAMI_MAPPING.keys())
CLASS_NAMES = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown']


def derive_features(rr_window: np.ndarray) -> np.ndarray:
    rr_window = rr_window.astype(float)
    rr_diff = np.diff(rr_window, prepend=rr_window[0])
    rolling = np.array([rr_window[max(0, i - 2):i + 1].mean() for i in range(len(rr_window))])
    rr_std = rr_window.std() if rr_window.std() > 1e-8 else 1.0
    rr_z = (rr_window - rr_window.mean()) / rr_std
    return np.stack([rr_window, rr_diff, rolling, rr_z], axis=-1)


def build_dataset(data_dir: str, n_steps: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    record_files = sorted(glob.glob(str(Path(data_dir) / '*.dat')))
    record_ids = [Path(f).stem for f in record_files]

    X, y, groups = [], [], []

    for record_id in record_ids:
        record_path = str(Path(data_dir) / record_id)

        try:
            ann = wfdb.rdann(record_path, 'atr')
            filtered_samples, filtered_symbols = [], []

            for sample, symbol in zip(ann.sample, ann.symbol):
                if symbol in VALID_BEATS:
                    filtered_samples.append(sample)
                    filtered_symbols.append(symbol)

            if len(filtered_samples) <= n_steps + 1:
                continue

            rr = np.diff(filtered_samples) / 360.0

            for i in range(len(rr) - n_steps):
                window = rr[i:i + n_steps]
                label = AAMI_MAPPING[filtered_symbols[i + n_steps + 1]]
                X.append(derive_features(window))
                y.append(label)
                groups.append(record_id)

        except Exception:
            continue

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(groups)
