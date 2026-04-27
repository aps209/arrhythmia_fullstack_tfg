"""
ecg_generator.py - Utilidades de renderizado para ECG real.

Se mantiene el nombre del módulo por compatibilidad, pero ya no genera
ECG sintético. Renderiza trazas reales y regiones destacadas.
"""

from __future__ import annotations

import base64
import io
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_ecg_preview(time_axis: Iterable[float], signal: Iterable[float], highlight_regions: Optional[list[dict]] = None) -> str:
    """Renderiza una vista previa PNG en base64 para un ECG real."""
    time_axis = np.asarray(list(time_axis), dtype=np.float32)
    signal = np.asarray(list(signal), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(14, 4), dpi=120)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    ax.plot(time_axis, signal, color="#19c0ff", linewidth=1.0, zorder=3)

    if highlight_regions:
        for region in highlight_regions:
            ax.axvspan(region["start_time"], region["end_time"], color="#ff6b6b", alpha=0.15, zorder=1)

    ax.grid(True, which="major", color="#1e2a44", linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", color="#1e2a44", linewidth=0.3, alpha=0.3)
    ax.minorticks_on()
    ax.set_xlabel("Tiempo (s)", color="#b7c0d8")
    ax.set_ylabel("Amplitud normalizada", color="#b7c0d8")
    ax.set_title("Vista previa ECG real", color="#ecf2ff", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#b7c0d8")
    for spine in ax.spines.values():
        spine.set_color("#2c3958")

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
