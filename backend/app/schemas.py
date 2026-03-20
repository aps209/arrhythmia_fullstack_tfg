"""
schemas.py — Modelos Pydantic para validación de entrada/salida de la API.

Define los contratos de datos para todos los endpoints:
  - Predicción de arritmias
  - Explicabilidad (XAI)
  - Generación de señal ECG
  - Información de la arquitectura del modelo
  - Pasos intermedios del pipeline
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional


# ─── Requests ──────────────────────────────────────────────────────────


class RRSequenceRequest(BaseModel):
    """Secuencia de 15 intervalos R-R en segundos."""
    rr_intervals: List[float] = Field(
        ...,
        min_length=15,
        max_length=15,
        description="15 intervalos R-R en segundos"
    )

    @field_validator("rr_intervals")
    @classmethod
    def validate_rr(cls, values: List[float]):
        for value in values:
            if value <= 0 or value > 5:
                raise ValueError(
                    "Cada intervalo R-R debe estar en segundos y dentro de un rango razonable (0, 5]."
                )
        return values


# ─── Responses ─────────────────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """Resultado de predicción con clase, confianza y nivel de riesgo."""
    predicted_class: int
    class_name: str
    confidence: float
    risk_level: str
    probabilities: Dict[str, float]
    input_summary: Dict[str, float]
    model_mode: str
    model_version: str


class ExplainResponse(BaseModel):
    """Resultado de explicabilidad con importancia por timestep."""
    predicted_class: int
    class_name: str
    confidence: float
    timestep_importance: List[float]
    top_timesteps: List[int]
    clinical_text: str
    model_mode: str
    model_version: str


class PredictExplainResponse(BaseModel):
    """Respuesta combinada de predicción + explicabilidad."""
    prediction: PredictionResponse
    explanation: ExplainResponse


class ECGSignalResponse(BaseModel):
    """Imagen ECG sintética codificada en base64 con metadata."""
    ecg_image_base64: str
    duration_seconds: float
    num_beats: int
    mean_heart_rate_bpm: float
    fs: int


class LayerInfo(BaseModel):
    """Información de una capa del modelo de red neuronal."""
    name: str
    type: str
    output_shape: Optional[str] = None
    num_params: int
    trainable: bool


class ModelArchitectureResponse(BaseModel):
    """Arquitectura completa del modelo incluyendo capas y metadatos."""
    model_mode: str
    total_params: int
    trainable_params: int
    layers: List[LayerInfo]
    input_shape: str
    output_shape: str
    optimizer: Optional[str] = None
    loss_function: Optional[str] = None


class FeatureStep(BaseModel):
    """Un paso de derivación de features con sus valores."""
    step_name: str
    description: str
    values: List[float]


class PipelineStepsResponse(BaseModel):
    """Visualización paso a paso del pipeline de procesamiento."""
    raw_rr: List[float]
    derived_features: List[FeatureStep]
    normalized_sample: Optional[List[List[float]]] = None
    prediction_probabilities: Dict[str, float]
    predicted_class: str
    model_mode: str
