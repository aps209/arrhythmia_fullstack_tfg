"""
schemas.py - Contratos de respuesta para la API basada en ECG real.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class PredictionResponse(APIModel):
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    analyzed_segments: int
    lead_name: str
    detection_source: str
    model_mode: str
    model_version: str
    model_key: str
    model_label: str


class HighlightRegion(APIModel):
    beat_index: int
    start_time: float
    end_time: float
    score: float


class WaveformPreview(APIModel):
    time: List[float]
    raw_signal: List[float]
    filtered_signal: List[float]
    highlight_regions: List[HighlightRegion]


class BeatPrediction(APIModel):
    beat_index: int
    sample: int
    time_seconds: float
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


class RepresentativeSegment(APIModel):
    beat_index: int
    signal: List[float]
    saliency: List[float]
    salient_region_ms: Dict[str, float]


class ExplainResponse(APIModel):
    predicted_class: int
    class_name: str
    confidence: float
    technical_summary: str
    evidence: List[str]
    limitations: List[str]
    detection_source: str
    top_beats: List[int]
    representative_segment: RepresentativeSegment
    model_mode: str
    model_version: str
    model_key: str
    model_label: str


class RecordSummary(APIModel):
    record_name: str
    original_fs: float
    target_fs: float
    duration_seconds: float
    num_samples: int
    analyzed_beats: int
    lead_name: str


class AnalyzeRecordResponse(APIModel):
    record: RecordSummary
    prediction: PredictionResponse
    explanation: ExplainResponse
    waveform_preview: WaveformPreview
    beat_predictions: List[BeatPrediction]


class PipelineStep(APIModel):
    step_name: str
    description: str
    details: Dict[str, float | int | str]


class PipelineStepsResponse(APIModel):
    record: RecordSummary
    preprocessing_steps: List[PipelineStep]
    waveform_preview: WaveformPreview
    representative_segment: Optional[RepresentativeSegment] = None
    prediction_probabilities: Dict[str, float]
    predicted_class: str
    model_mode: str
    model_key: str
    model_label: str


class LayerInfo(APIModel):
    name: str
    type: str
    output_shape: Optional[str] = None
    num_params: int
    trainable: bool


class ModelArchitectureResponse(APIModel):
    model_key: str
    model_label: str
    model_mode: str
    architecture_name: Optional[str] = None
    total_params: int
    trainable_params: int
    layers: List[LayerInfo]
    input_shape: Optional[str] = None
    output_shape: Optional[str] = None
    optimizer: Optional[str] = None
    loss_function: Optional[str] = None
