"""
wfdb_tools.py - Utilidades para cargar, preprocesar y segmentar ECG WFDB.

El backend y el pipeline de entrenamiento comparten estas funciones para:
  - leer registros MIT-BIH / WFDB reales (.hea + .dat)
  - seleccionar una derivación útil para clasificación
  - filtrar y remuestrear la señal
  - recuperar anotaciones o detectar picos R si no existe .atr
  - extraer ventanas centradas en latidos para el modelo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import wfdb
from scipy import signal


AAMI_MAPPING = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}

CLASS_NAMES = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"]
VALID_BEATS = set(AAMI_MAPPING.keys())
PREFERRED_LEADS = ("MLII", "II", "V5", "V1")


@dataclass
class LoadedRecord:
    record_name: str
    original_fs: float
    target_fs: float
    lead_name: str
    signal: np.ndarray
    filtered_signal: np.ndarray


@dataclass
class AnnotationBundle:
    samples: np.ndarray
    symbols: List[str]
    source: str


def bandpass_filter(ecg: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """Aplica un filtro Butterworth pasabanda para eliminar deriva y ruido de alta frecuencia."""
    ecg = np.asarray(ecg, dtype=np.float32)
    if ecg.size == 0:
        return ecg

    nyquist = fs / 2.0
    highcut = min(highcut, nyquist - 0.5) if nyquist > 1 else highcut
    if highcut <= lowcut:
        return ecg

    b, a = signal.butter(3, [lowcut / nyquist, highcut / nyquist], btype="band")
    return signal.filtfilt(b, a, ecg).astype(np.float32)


def normalize_segment(segment: np.ndarray) -> np.ndarray:
    """Normaliza una ventana ECG con z-score por segmento."""
    segment = np.asarray(segment, dtype=np.float32)
    std = float(segment.std())
    if std < 1e-6:
        std = 1.0
    return ((segment - float(segment.mean())) / std).astype(np.float32)


def select_lead(signal_matrix: np.ndarray, lead_names: Iterable[str], preferred_lead: Optional[str] = None) -> tuple[np.ndarray, str]:
    """Selecciona una derivación priorizando MLII/II/V5/V1 o la solicitada por el usuario."""
    names = [str(name) for name in lead_names]
    signal_matrix = np.asarray(signal_matrix, dtype=np.float32)
    normalized_names = {name.upper(): idx for idx, name in enumerate(names)}

    if preferred_lead:
        preferred_idx = normalized_names.get(preferred_lead.upper())
        if preferred_idx is not None:
            return signal_matrix[:, preferred_idx], names[preferred_idx]

    for candidate in PREFERRED_LEADS:
        idx = normalized_names.get(candidate)
        if idx is not None:
            return signal_matrix[:, idx], names[idx]

    return signal_matrix[:, 0], names[0] if names else "lead_0"


def resample_signal(ecg: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Remuestrea la señal al `target_fs` si es necesario."""
    ecg = np.asarray(ecg, dtype=np.float32)
    if abs(original_fs - target_fs) < 1e-6:
        return ecg

    gcd = np.gcd(int(round(original_fs)), int(round(target_fs)))
    up = int(round(target_fs / gcd))
    down = int(round(original_fs / gcd))
    return signal.resample_poly(ecg, up, down).astype(np.float32)


def load_record(record_path: Path, target_fs: float, preferred_lead: Optional[str] = None) -> LoadedRecord:
    """Carga un registro WFDB, selecciona derivación, remuestrea y filtra la señal."""
    wfdb_record = wfdb.rdrecord(str(record_path))
    if wfdb_record.p_signal is None:
        raise ValueError("El registro WFDB no contiene `p_signal` disponible.")

    lead_signal, lead_name = select_lead(wfdb_record.p_signal, wfdb_record.sig_name, preferred_lead)
    original_fs = float(wfdb_record.fs)
    resampled_signal = resample_signal(lead_signal, original_fs, target_fs)
    filtered_signal = bandpass_filter(resampled_signal, target_fs)

    return LoadedRecord(
        record_name=record_path.name,
        original_fs=original_fs,
        target_fs=float(target_fs),
        lead_name=lead_name,
        signal=resampled_signal,
        filtered_signal=filtered_signal,
    )


def load_annotations(record_path: Path, original_fs: float, target_fs: float) -> Optional[AnnotationBundle]:
    """Carga y remapea anotaciones AAMI si existe un fichero .atr."""
    atr_path = record_path.with_suffix(".atr")
    if not atr_path.exists():
        return None

    ann = wfdb.rdann(str(record_path), "atr")
    samples: List[int] = []
    symbols: List[str] = []

    scale = float(target_fs) / float(original_fs)
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in VALID_BEATS:
            continue
        samples.append(int(round(float(sample) * scale)))
        symbols.append(symbol)

    if not samples:
        return None

    return AnnotationBundle(
        samples=np.asarray(samples, dtype=np.int32),
        symbols=symbols,
        source="annotations",
    )


def detect_r_peaks(filtered_signal: np.ndarray, fs: float) -> AnnotationBundle:
    """Detecta picos R con una heurística robusta si no existe .atr."""
    rectified = np.abs(filtered_signal)
    distance = max(1, int(0.25 * fs))
    prominence = max(0.15, float(np.percentile(rectified, 90) * 0.35))
    peaks, _ = signal.find_peaks(rectified, distance=distance, prominence=prominence)

    return AnnotationBundle(
        samples=np.asarray(peaks, dtype=np.int32),
        symbols=["?"] * len(peaks),
        source="detected_peaks",
    )


def segment_windows(
    filtered_signal: np.ndarray,
    centers: np.ndarray,
    window_size: int,
    pre_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrae ventanas centradas en latidos y las normaliza para el modelo."""
    filtered_signal = np.asarray(filtered_signal, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.int32)

    segments: List[np.ndarray] = []
    valid_centers: List[int] = []
    post_samples = window_size - pre_samples

    for center in centers:
        start = int(center) - pre_samples
        end = int(center) + post_samples
        if start < 0 or end > len(filtered_signal):
            continue
        segment = filtered_signal[start:end]
        if len(segment) != window_size:
            continue
        segments.append(normalize_segment(segment))
        valid_centers.append(int(center))

    if not segments:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.stack(segments).astype(np.float32), np.asarray(valid_centers, dtype=np.int32)


def build_highlight_regions(centers: np.ndarray, fs: float, pre_samples: int, post_samples: int, scores: np.ndarray) -> list[dict]:
    """Construye regiones temporales resaltables para frontend."""
    regions = []
    for idx, (center, score) in enumerate(zip(centers, scores), start=1):
        regions.append(
            {
                "beat_index": idx,
                "start_time": round((int(center) - pre_samples) / fs, 4),
                "end_time": round((int(center) + post_samples) / fs, 4),
                "score": round(float(score), 6),
            }
        )
    return regions


def downsample_for_preview(time_axis: np.ndarray, raw_signal: np.ndarray, filtered_signal: np.ndarray, max_points: int = 4000) -> dict:
    """Reduce puntos para visualización sin devolver registros completos enormes."""
    time_axis = np.asarray(time_axis, dtype=np.float32)
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    filtered_signal = np.asarray(filtered_signal, dtype=np.float32)

    if len(time_axis) <= max_points:
        step = 1
    else:
        step = int(np.ceil(len(time_axis) / max_points))

    return {
        "time": np.round(time_axis[::step], 5).tolist(),
        "raw_signal": np.round(raw_signal[::step], 6).tolist(),
        "filtered_signal": np.round(filtered_signal[::step], 6).tolist(),
    }


def contiguous_region_from_saliency(saliency: np.ndarray, fs: float) -> dict:
    """Resume la región más saliente de un mapa temporal."""
    saliency = np.asarray(saliency, dtype=np.float32)
    if saliency.size == 0:
        return {"start_ms": 0.0, "end_ms": 0.0, "width_ms": 0.0}

    threshold = float(np.quantile(saliency, 0.85))
    mask = saliency >= threshold
    if not mask.any():
        peak = int(np.argmax(saliency))
        return {
            "start_ms": round((peak / fs) * 1000.0, 2),
            "end_ms": round((peak / fs) * 1000.0, 2),
            "width_ms": 0.0,
        }

    indices = np.where(mask)[0]
    start = int(indices[0])
    end = int(indices[-1])
    return {
        "start_ms": round((start / fs) * 1000.0, 2),
        "end_ms": round((end / fs) * 1000.0, 2),
        "width_ms": round(((end - start + 1) / fs) * 1000.0, 2),
    }
