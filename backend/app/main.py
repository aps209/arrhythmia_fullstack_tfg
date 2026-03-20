"""
main.py — Aplicación FastAPI principal para la API de predicción de arritmias.

Endpoints disponibles:
  GET  /health               → Estado del sistema
  GET  /model-info            → Información del modelo
  GET  /model-architecture    → Arquitectura detallada de la red neuronal
  POST /predict               → Predicción de arritmia
  POST /explain               → Explicabilidad por oclusión
  POST /predict-and-explain   → Predicción + explicabilidad combinadas
  POST /ecg-signal            → Generación de señal ECG sintética
  POST /pipeline-steps        → Pasos intermedios del pipeline
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    RRSequenceRequest,
    PredictionResponse,
    ExplainResponse,
    PredictExplainResponse,
    ECGSignalResponse,
    ModelArchitectureResponse,
    PipelineStepsResponse,
)
from .model_service import service
from .ecg_generator import generate_ecg_image

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Aplicación FastAPI ────────────────────────────────────────────────

app = FastAPI(
    title="Arrhythmia Early Warning API",
    description=(
        "API REST para predicción temprana de arritmias cardíacas "
        "a partir de ventanas de 15 intervalos R-R. Incluye explicabilidad "
        "por oclusión, generación de ECG sintético y visualización del pipeline."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints de estado ──────────────────────────────────────────────


@app.get("/health", tags=["Sistema"])
def health():
    """Comprueba el estado de salud del servicio y el modelo cargado."""
    return {
        "status": "ok",
        "model_mode": service.mode,
        "model_version": service.metadata.get("model_version", "unknown"),
        "expected_input": "15_rr_intervals_seconds",
    }


@app.get("/model-info", tags=["Sistema"])
def model_info():
    """Devuelve información general del modelo y su metadata."""
    return {
        "model_mode": service.mode,
        "metadata": service.metadata,
    }


# ─── Endpoints de predicción ──────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(payload: RRSequenceRequest):
    """
    Predice la clase de arritmia a partir de 15 intervalos R-R.
    Devuelve clase predicha, confianza, nivel de riesgo y probabilidades.
    """
    logger.info("POST /predict — %d intervalos recibidos", len(payload.rr_intervals))
    return service.make_prediction_response(payload.rr_intervals)


@app.post("/explain", response_model=ExplainResponse, tags=["Explicabilidad"])
def explain(payload: RRSequenceRequest):
    """
    Genera explicación de la predicción identificando los timesteps
    más influyentes mediante oclusión.
    """
    logger.info("POST /explain — análisis de explicabilidad")
    return service.explain(payload.rr_intervals)


@app.post("/predict-and-explain", response_model=PredictExplainResponse, tags=["Predicción"])
def predict_and_explain(payload: RRSequenceRequest):
    """Predicción y explicabilidad combinadas en una sola llamada."""
    logger.info("POST /predict-and-explain")
    prediction = service.make_prediction_response(payload.rr_intervals)
    explanation = service.explain(payload.rr_intervals)
    return {"prediction": prediction, "explanation": explanation}


# ─── Endpoints de visualización ────────────────────────────────────────


@app.post("/ecg-signal", response_model=ECGSignalResponse, tags=["Visualización"])
def ecg_signal(payload: RRSequenceRequest):
    """
    Genera una señal ECG sintética a partir de los intervalos R-R
    y devuelve la imagen como PNG codificado en base64.
    """
    logger.info("POST /ecg-signal — generando ECG sintético")
    try:
        rr = payload.rr_intervals
        explanation = service.explain(rr)
        highlight = explanation.get("top_timesteps", [])

        ecg_b64 = generate_ecg_image(rr, highlight_indices=highlight)
        duration = sum(rr)
        mean_hr = 60.0 / (sum(rr) / len(rr)) if sum(rr) > 0 else 0.0

        return {
            "ecg_image_base64": ecg_b64,
            "duration_seconds": round(duration, 3),
            "num_beats": len(rr),
            "mean_heart_rate_bpm": round(mean_hr, 1),
            "fs": 500,
        }
    except Exception as e:
        logger.error("Error generando ECG: %s", e)
        raise HTTPException(status_code=500, detail=f"Error al generar ECG: {str(e)}")


@app.get("/model-architecture", response_model=ModelArchitectureResponse, tags=["Visualización"])
def model_architecture():
    """
    Devuelve la arquitectura completa del modelo: capas, parámetros,
    shapes de entrada/salida y configuración del optimizador.
    """
    logger.info("GET /model-architecture")
    return service.get_architecture_info()


@app.post("/pipeline-steps", response_model=PipelineStepsResponse, tags=["Visualización"])
def pipeline_steps(payload: RRSequenceRequest):
    """
    Muestra paso a paso cómo el pipeline transforma los R-R crudos
    a través de la derivación de features, normalización y predicción.
    Diseñado para uso educativo y visualización en el frontend.
    """
    logger.info("POST /pipeline-steps — visualización del pipeline")
    return service.get_pipeline_steps(payload.rr_intervals)
