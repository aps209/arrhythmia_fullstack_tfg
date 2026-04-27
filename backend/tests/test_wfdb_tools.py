import numpy as np

from backend.app.wfdb_tools import bandpass_filter, contiguous_region_from_saliency, normalize_segment, segment_windows


def test_normalize_segment_zero_mean_unit_scale():
    segment = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    normalized = normalize_segment(segment)
    assert abs(float(normalized.mean())) < 1e-5
    assert abs(float(normalized.std()) - 1.0) < 1e-5


def test_segment_windows_respects_boundaries():
    signal = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
    centers = np.array([10, 100, 190], dtype=np.int32)
    segments, valid_centers = segment_windows(signal, centers, window_size=40, pre_samples=15)
    assert segments.shape == (1, 40)
    assert valid_centers.tolist() == [100]


def test_bandpass_filter_preserves_length():
    time = np.linspace(0, 2, 500, endpoint=False)
    ecg = np.sin(2 * np.pi * 5 * time).astype(np.float32)
    filtered = bandpass_filter(ecg, fs=250)
    assert len(filtered) == len(ecg)


def test_contiguous_region_from_saliency():
    saliency = np.array([0.0, 0.2, 0.95, 0.9, 0.1], dtype=np.float32)
    region = contiguous_region_from_saliency(saliency, fs=250)
    assert region["start_ms"] <= region["end_ms"]
    assert region["width_ms"] >= 0
