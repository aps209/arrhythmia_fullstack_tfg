"""
main.py - API FastAPI para analisis de ECG real mediante WFDB.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import DEFAULT_MODEL_KEY
from .model_service import ModelUnavailableError, service
from .schemas import AnalyzeRecordResponse, ModelArchitectureResponse, PipelineStepsResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Arrhythmia ECG Analysis API",
    description=(
        "API REST para analisis de arritmias a partir de ECG crudo en formato WFDB "
        "(.dat + .hea y opcionalmente .atr)."
    ),
    version="4.0.0",
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


def _persist_uploads(temp_dir: Path, files: List[UploadFile]) -> Path:
    stems = {}
    for upload in files:
        filename = Path(upload.filename or "")
        if not filename.name:
            continue
        if filename.suffix.lower() not in {".dat", ".hea", ".atr"}:
            continue
        destination = temp_dir / filename.name
        with destination.open("wb") as buffer:
            buffer.write(upload.file.read())
        stems.setdefault(filename.stem, set()).add(filename.suffix.lower())

    candidates = [stem for stem, suffixes in stems.items() if ".dat" in suffixes and ".hea" in suffixes]
    if not candidates:
        raise HTTPException(status_code=422, detail="Debes subir al menos un par `.dat + .hea` del mismo registro.")

    return temp_dir / sorted(candidates)[0]


def _handle_analysis(record_path: Path, model_key: Optional[str] = None) -> dict:
    try:
        return service.analyze_record(record_path, preferred_lead=None, model_key=model_key)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error analizando el registro WFDB")
        raise HTTPException(status_code=500, detail=f"Error interno analizando ECG: {exc}") from exc


@app.get("/health", tags=["Sistema"])
def health():
    default_info = service.get_model_info(DEFAULT_MODEL_KEY)
    return {
        "status": "ok",
        "model_mode": service.mode,
        "model_version": service.metadata.get("model_version", "unknown"),
        "default_model_key": DEFAULT_MODEL_KEY,
        "available_models": service.available_models(),
        "expected_input": "wfdb_bundle_dat_hea_optional_atr",
        "selected_model": {
            "model_key": default_info["model_key"],
            "model_label": default_info["model_label"],
        },
    }


@app.get("/model-info", tags=["Sistema"])
def model_info(model_key: Optional[str] = Query(default=None)):
    return service.get_model_info(model_key=model_key)


@app.get("/model-architecture", response_model=ModelArchitectureResponse, tags=["Sistema"])
def model_architecture(model_key: Optional[str] = Query(default=None)):
    return service.get_architecture_info(model_key=model_key)


@app.post("/analyze-record", response_model=AnalyzeRecordResponse, tags=["Inferencia"])
def analyze_record(
    files: List[UploadFile] = File(...),
    model_key: Optional[str] = Form(default=None),
):
    logger.info("POST /analyze-record - %d archivos recibidos - modelo=%s", len(files), model_key or DEFAULT_MODEL_KEY)
    with tempfile.TemporaryDirectory() as tmp_dir:
        record_path = _persist_uploads(Path(tmp_dir), files)
        return _handle_analysis(record_path, model_key=model_key)


@app.post("/predict-and-explain", response_model=AnalyzeRecordResponse, tags=["Inferencia"])
def predict_and_explain(
    files: List[UploadFile] = File(...),
    model_key: Optional[str] = Form(default=None),
):
    logger.info("POST /predict-and-explain - alias de /analyze-record")
    with tempfile.TemporaryDirectory() as tmp_dir:
        record_path = _persist_uploads(Path(tmp_dir), files)
        return _handle_analysis(record_path, model_key=model_key)


@app.post("/pipeline-steps", response_model=PipelineStepsResponse, tags=["Visualizacion"])
def pipeline_steps(
    files: List[UploadFile] = File(...),
    model_key: Optional[str] = Form(default=None),
):
    logger.info("POST /pipeline-steps - %d archivos recibidos - modelo=%s", len(files), model_key or DEFAULT_MODEL_KEY)
    with tempfile.TemporaryDirectory() as tmp_dir:
        record_path = _persist_uploads(Path(tmp_dir), files)
        try:
            return service.get_pipeline_steps(record_path, preferred_lead=None, model_key=model_key)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error obteniendo pipeline")
            raise HTTPException(status_code=500, detail=f"Error interno obteniendo pipeline: {exc}") from exc
