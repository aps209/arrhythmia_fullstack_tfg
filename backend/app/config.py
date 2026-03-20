"""
config.py — Configuración centralizada del backend.

Define rutas a artefactos del modelo, nombres de clases AAMI
y parámetros de la ventana temporal.
"""

from pathlib import Path
import os

# Directorio de artefactos del modelo (model.keras, scaler.joblib, metadata.json)
# Prioriza ARTIFACTS_PATH (Docker), luego MODEL_DIR, luego directorio local
_artifacts_env = os.getenv("ARTIFACTS_PATH", os.getenv("MODEL_DIR", ""))
if _artifacts_env:
    MODEL_DIR = Path(_artifacts_env)
else:
    MODEL_DIR = Path(__file__).resolve().parents[2] / "artifacts"

MODEL_PATH = MODEL_DIR / "model.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Mapeo de clases según estándar AAMI EC57
CLASS_NAMES = {
    0: "Normal",
    1: "SVEB",
    2: "VEB",
    3: "Fusion",
    4: "Unknown",
}

# Longitud de la ventana de intervalos R-R
N_STEPS = 15
