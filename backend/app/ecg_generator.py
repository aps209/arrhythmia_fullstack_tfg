"""
ecg_generator.py — Generador de señal ECG sintética.

Genera una señal ECG realista a partir de intervalos R-R usando ondas
PQRST modeladas con funciones gaussianas. La señal resultante se visualiza
como imagen PNG codificada en base64 para su uso en el frontend.
"""

import base64
import io
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend no-interactivo para servidores
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ─── Parámetros de las ondas PQRST ────────────────────────────────────
# Cada onda se define como: (amplitud_relativa, centro_relativo, anchura)
# centro_relativo es la posición dentro de un ciclo RR normalizado [0, 1]
WAVE_PARAMS = {
    "P": {"amplitude": 0.15, "center": 0.12, "width": 0.04},
    "Q": {"amplitude": -0.10, "center": 0.22, "width": 0.012},
    "R": {"amplitude": 1.00, "center": 0.25, "width": 0.018},
    "S": {"amplitude": -0.20, "center": 0.28, "width": 0.012},
    "T": {"amplitude": 0.30, "center": 0.42, "width": 0.055},
}


def _gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Genera una onda gaussiana centrada en `center` con la amplitud y anchura dadas."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


def synthesize_ecg_signal(
    rr_intervals: List[float],
    fs: int = 500,
    noise_level: float = 0.02,
) -> Dict[str, Any]:
    """
    Genera una señal ECG sintética a partir de intervalos R-R.

    Parameters
    ----------
    rr_intervals : list[float]
        Intervalos R-R en segundos.
    fs : int
        Frecuencia de muestreo (Hz). Default: 500.
    noise_level : float
        Amplitud del ruido gaussiano. Default: 0.02.

    Returns
    -------
    dict con claves:
        - signal: np.ndarray con la señal ECG
        - time: np.ndarray con el eje temporal
        - r_peak_positions: list[float] posiciones temporales de los picos R
        - fs: frecuencia de muestreo
        - duration: duración total en segundos
    """
    rr = np.array(rr_intervals, dtype=float)
    total_duration = float(rr.sum())
    n_samples = int(total_duration * fs)
    time = np.linspace(0, total_duration, n_samples, endpoint=False)
    signal = np.zeros(n_samples)
    r_peak_positions = []

    cumulative_time = 0.0
    for interval in rr:
        cycle_start = cumulative_time
        cycle_end = cumulative_time + interval
        r_peak_positions.append(cycle_start + WAVE_PARAMS["R"]["center"] * interval)

        # Máscara de muestras dentro de este ciclo
        mask = (time >= cycle_start) & (time < cycle_end)
        t_local = (time[mask] - cycle_start) / interval  # Normalizado [0, 1)

        for wave_name, params in WAVE_PARAMS.items():
            signal[mask] += _gaussian(t_local, params["amplitude"], params["center"], params["width"])

        cumulative_time = cycle_end

    # Añadir ruido realista
    signal += np.random.normal(0, noise_level, n_samples)

    # Leve deriva de línea base (baseline wander)
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.15 * time)
    signal += baseline_wander

    return {
        "signal": signal,
        "time": time,
        "r_peak_positions": r_peak_positions,
        "fs": fs,
        "duration": total_duration,
    }


def generate_ecg_image(
    rr_intervals: List[float],
    highlight_indices: List[int] = None,
    figsize: tuple = (14, 5),
    dpi: int = 120,
) -> str:
    """
    Genera la imagen PNG de un ECG sintético y la devuelve codificada en base64.

    Parameters
    ----------
    rr_intervals : list[float]
        15 intervalos R-R en segundos.
    highlight_indices : list[int], optional
        Índices (1-based) de los intervalos R-R a destacar.
    figsize : tuple
        Tamaño de la figura.
    dpi : int
        Resolución.

    Returns
    -------
    str : imagen PNG codificada en base64.
    """
    ecg_data = synthesize_ecg_signal(rr_intervals)
    signal = ecg_data["signal"]
    time = ecg_data["time"]
    r_peaks = ecg_data["r_peak_positions"]

    # Estilo oscuro profesional
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Fondo con gradiente simulado
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Cuadrícula tipo papel ECG
    ax.grid(True, which="major", color="#1a2332", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", color="#1a2332", linewidth=0.3, alpha=0.3)
    ax.minorticks_on()

    # Señal ECG principal
    ax.plot(time, signal, color="#00d4ff", linewidth=1.0, alpha=0.95, zorder=3)

    # Marcar picos R
    for i, r_pos in enumerate(r_peaks):
        idx = np.argmin(np.abs(time - r_pos))
        ax.plot(r_pos, signal[idx], "v", color="#ff6b6b", markersize=8, zorder=5, alpha=0.8)

    # Destacar intervalos R-R importantes
    if highlight_indices:
        cumulative = 0.0
        rr = np.array(rr_intervals)
        for i in range(len(rr)):
            if (i + 1) in highlight_indices:
                ax.axvspan(cumulative, cumulative + rr[i],
                          color="#ff6b6b", alpha=0.12, zorder=1)
            cumulative += rr[i]

    # Etiquetas
    ax.set_xlabel("Tiempo (s)", fontsize=12, fontfamily="sans-serif", color="#8b949e")
    ax.set_ylabel("Amplitud (mV)", fontsize=12, fontfamily="sans-serif", color="#8b949e")
    ax.set_title("Electrocardiograma Sintético — Derivación II",
                fontsize=14, fontweight="bold", fontfamily="sans-serif",
                color="#e6edf3", pad=15)

    ax.tick_params(colors="#8b949e", labelsize=10)

    for spine in ax.spines.values():
        spine.set_color("#30363d")
        spine.set_linewidth(0.5)

    plt.tight_layout()

    # Exportar a base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
