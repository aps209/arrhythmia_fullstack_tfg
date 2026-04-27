"""
model_service.py - Servicio de inferencia y explicabilidad sobre ECG crudo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from .config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_PRE_SAMPLES,
    DEFAULT_TARGET_FS,
    DEFAULT_WINDOW_SIZE,
    MODEL_SPECS,
    default_metadata_for,
    model_dir_for,
    model_paths_for,
)
from .wfdb_tools import (
    CLASS_NAMES,
    build_highlight_regions,
    contiguous_region_from_saliency,
    detect_r_peaks,
    downsample_for_preview,
    load_annotations,
    load_record,
    segment_windows,
)

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Se lanza cuando el backend no tiene artefactos validos cargados."""


class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
    """Compatibilidad con modelos serializados que incluyen opciones legacy de BatchNormalization."""

    @classmethod
    def from_config(cls, config: dict) -> "CompatibleBatchNormalization":
        config = dict(config)
        config.pop("renorm", None)
        config.pop("renorm_clipping", None)
        config.pop("renorm_momentum", None)
        return cls(**config)


@dataclass
class LoadedModel:
    key: str
    label: str
    metadata: Dict
    model: Optional[tf.keras.Model]
    mode: str
    model_dir: Path
    model_path: Path
    metadata_path: Path


class ModelService:
    def __init__(self):
        self.models: Dict[str, LoadedModel] = {}
        self._load()

    def _load(self) -> None:
        self.models = {}
        for model_key, spec in MODEL_SPECS.items():
            model_path, metadata_path = model_paths_for(model_key)
            metadata = default_metadata_for(model_key)
            metadata["model_dir"] = str(model_dir_for(model_key))

            if metadata_path.exists():
                try:
                    metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
                except Exception as exc:
                    logger.error("No se pudo cargar metadata de %s: %s", model_key, exc)

            model = None
            mode = "unavailable"
            if model_path.exists():
                try:
                    model = load_model(
                        model_path,
                        compile=False,
                        custom_objects={"BatchNormalization": CompatibleBatchNormalization},
                    )
                    mode = "trained"
                except Exception as exc:
                    logger.error("No se pudo cargar el modelo %s: %s", model_key, exc)

            metadata["model_key"] = model_key
            metadata["model_label"] = spec["label"]
            metadata["architecture_name"] = metadata.get("architecture_name") or spec["architecture_name"]

            self.models[model_key] = LoadedModel(
                key=model_key,
                label=spec["label"],
                metadata=metadata,
                model=model,
                mode=mode,
                model_dir=model_dir_for(model_key),
                model_path=model_path,
                metadata_path=metadata_path,
            )

    @property
    def default_model(self) -> LoadedModel:
        return self.models[DEFAULT_MODEL_KEY]

    @property
    def mode(self) -> str:
        return self.default_model.mode

    @property
    def metadata(self) -> Dict:
        return self.default_model.metadata

    def available_models(self) -> list[dict]:
        return [
            {
                "model_key": model.key,
                "model_label": model.label,
                "architecture_name": model.metadata.get("architecture_name"),
                "model_mode": model.mode,
                "model_version": model.metadata.get("model_version", "unknown"),
                "target_fs": int(model.metadata.get("target_fs", DEFAULT_TARGET_FS)),
                "window_size": int(model.metadata.get("window_size", DEFAULT_WINDOW_SIZE)),
            }
            for model in self.models.values()
        ]

    def _get_model(self, model_key: Optional[str] = None) -> LoadedModel:
        selected_key = model_key or DEFAULT_MODEL_KEY
        if selected_key not in self.models:
            valid = ", ".join(self.models.keys())
            raise ValueError(f"Modelo desconocido '{selected_key}'. Modelos validos: {valid}.")
        return self.models[selected_key]

    def _require_model(self, loaded_model: LoadedModel) -> None:
        if loaded_model.model is None or loaded_model.mode != "trained":
            raise ModelUnavailableError(
                f"No hay un modelo entrenado disponible para '{loaded_model.key}'. "
                f"Entrena el modelo y genera {loaded_model.model_path} + {loaded_model.metadata_path}."
            )

    def _target_fs(self, loaded_model: LoadedModel) -> int:
        return int(loaded_model.metadata.get("target_fs", DEFAULT_TARGET_FS))

    def _window_size(self, loaded_model: LoadedModel) -> int:
        return int(loaded_model.metadata.get("window_size", DEFAULT_WINDOW_SIZE))

    def _pre_samples(self, loaded_model: LoadedModel) -> int:
        return int(loaded_model.metadata.get("pre_samples", DEFAULT_PRE_SAMPLES))

    def _post_samples(self, loaded_model: LoadedModel) -> int:
        return int(self._window_size(loaded_model) - self._pre_samples(loaded_model))

    def _predict_segments(self, loaded_model: LoadedModel, segments: np.ndarray) -> np.ndarray:
        self._require_model(loaded_model)
        assert loaded_model.model is not None
        if segments.ndim == 2:
            segments = segments[..., np.newaxis]
        probs = loaded_model.model.predict(segments.astype(np.float32), verbose=0)
        return np.asarray(probs, dtype=np.float32)

    def _compute_saliency(self, loaded_model: LoadedModel, segment: np.ndarray, class_idx: int) -> np.ndarray:
        self._require_model(loaded_model)
        assert loaded_model.model is not None
        tensor = tf.convert_to_tensor(segment[np.newaxis, :, np.newaxis], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            probs = loaded_model.model(tensor, training=False)
            score = probs[:, class_idx]
        grads = tape.gradient(score, tensor)
        saliency = np.abs(grads.numpy()[0, :, 0]).astype(np.float32)
        max_val = float(saliency.max())
        if max_val > 0:
            saliency /= max_val
        return saliency

    def _qrs_width_ms(self, segment: np.ndarray, fs: float, pre_samples: int) -> float:
        center = pre_samples
        left = center
        right = center
        threshold = max(0.15, float(np.abs(segment[center]) * 0.5))

        while left > 0 and abs(segment[left]) > threshold:
            left -= 1
        while right < len(segment) - 1 and abs(segment[right]) > threshold:
            right += 1

        return round(((right - left + 1) / fs) * 1000.0, 2)

    def _build_explanation(
        self,
        loaded_model: LoadedModel,
        segments: np.ndarray,
        centers: np.ndarray,
        mean_probs: np.ndarray,
        segment_probs: np.ndarray,
        fs: float,
        detection_source: str,
    ) -> Dict:
        pred_idx = int(np.argmax(mean_probs))
        pred_name = CLASS_NAMES[pred_idx]
        pre_samples = self._pre_samples(loaded_model)

        segment_scores = segment_probs[:, pred_idx]
        ranked = np.argsort(segment_scores)[::-1]
        top_segment_idx = int(ranked[0])
        top_beats = [int(i + 1) for i in ranked[:3]]

        representative = segments[top_segment_idx]
        saliency = self._compute_saliency(loaded_model, representative, pred_idx)
        salient_region = contiguous_region_from_saliency(saliency, fs)
        qrs_width = self._qrs_width_ms(representative, fs, pre_samples=pre_samples)
        peak_to_peak = round(float(representative.max() - representative.min()), 4)
        max_upstroke = round(float(np.max(np.abs(np.diff(representative)))), 4)
        confidence = float(mean_probs[pred_idx])
        architecture_name = loaded_model.metadata.get("architecture_name", loaded_model.label)

        class_specific = {
            "Normal": (
                "El patron dominante es compatible con morfologia sinusal relativamente estable, "
                "con dispersion temporal baja y activacion concentrada en un complejo estrecho."
            ),
            "SVEB": (
                "La red prioriza segmentos con alteracion supraventricular, con desviacion temporal precoz "
                "y redistribucion de energia antes del complejo principal."
            ),
            "VEB": (
                "La decision esta dominada por complejos de morfologia aberrante y energia extendida, "
                "consistente con despolarizacion ventricular ectopica."
            ),
            "Fusion": (
                "La clase fusion aparece cuando la red detecta mezcla de componentes morfologicos normales y ventriculares "
                "en un mismo latido o en latidos consecutivos cercanos."
            ),
            "Unknown": (
                "La distribucion de probabilidad sigue siendo difusa y la morfologia no encaja de forma estable en una superclase AAMI."
            ),
        }

        technical_summary = (
            f"Modelo evaluado: {loaded_model.label}. Clasificacion agregada: {pred_name} con confianza media {confidence:.2%}. "
            f"{class_specific[pred_name]} En el segmento representativo, la region mas influyente ocupa "
            f"{salient_region['width_ms']:.2f} ms y el ancho QRS estimado es {qrs_width:.2f} ms."
        )

        evidence = [
            f"Se analizaron {len(segments)} latidos segmentados a {fs:.0f} Hz con agregacion por media de probabilidades.",
            f"Los latidos con mayor evidencia para {pred_name} fueron {', '.join(map(str, top_beats))}.",
            f"Segmento representativo: pico a pico {peak_to_peak:.4f}, pendiente maxima {max_upstroke:.4f}, region saliente entre {salient_region['start_ms']:.2f} y {salient_region['end_ms']:.2f} ms.",
            f"Arquitectura usada en inferencia: {architecture_name}.",
        ]

        limitations = [
            "El analisis se realiza sobre una unica derivacion seleccionada automaticamente.",
            "La explicabilidad basada en gradientes identifica sensibilidad del modelo, no causalidad fisiologica directa.",
            "La interpretacion depende de la calidad del registro, del remuestreo y de la deteccion/centrado de latidos.",
        ]
        if detection_source != "annotations":
            limitations.append("No habia anotaciones .atr; el centrado de latidos se hizo con deteccion automatica de picos R.")

        return {
            "predicted_class": pred_idx,
            "class_name": pred_name,
            "confidence": confidence,
            "technical_summary": technical_summary,
            "evidence": evidence,
            "limitations": limitations,
            "detection_source": detection_source,
            "top_beats": top_beats,
            "representative_segment": {
                "beat_index": top_segment_idx + 1,
                "signal": np.round(representative, 6).tolist(),
                "saliency": np.round(saliency, 6).tolist(),
                "salient_region_ms": salient_region,
            },
            "model_mode": loaded_model.mode,
            "model_version": loaded_model.metadata.get("model_version", "unknown"),
            "model_key": loaded_model.key,
            "model_label": loaded_model.label,
        }

    def _build_record_summary(
        self,
        record_name: str,
        original_fs: float,
        target_fs: float,
        signal: np.ndarray,
        analyzed_beats: int,
        lead_name: str,
    ) -> Dict:
        return {
            "record_name": record_name,
            "original_fs": float(original_fs),
            "target_fs": float(target_fs),
            "duration_seconds": round(len(signal) / target_fs, 3),
            "num_samples": int(len(signal)),
            "analyzed_beats": int(analyzed_beats),
            "lead_name": lead_name,
        }

    def analyze_record(
        self,
        record_path: Path,
        preferred_lead: Optional[str] = None,
        model_key: Optional[str] = None,
    ) -> Dict:
        loaded_model = self._get_model(model_key)
        self._require_model(loaded_model)

        target_fs = self._target_fs(loaded_model)
        pre_samples = self._pre_samples(loaded_model)
        window_size = self._window_size(loaded_model)
        post_samples = self._post_samples(loaded_model)

        loaded = load_record(record_path, target_fs=target_fs, preferred_lead=preferred_lead)
        annotations = load_annotations(record_path, original_fs=loaded.original_fs, target_fs=loaded.target_fs)
        if annotations is None:
            annotations = detect_r_peaks(loaded.filtered_signal, loaded.target_fs)

        segments, centers = segment_windows(
            loaded.filtered_signal,
            annotations.samples,
            window_size=window_size,
            pre_samples=pre_samples,
        )
        if len(segments) == 0:
            raise ValueError("No se pudieron extraer ventanas validas del ECG.")

        segment_probs = self._predict_segments(loaded_model, segments)
        mean_probs = segment_probs.mean(axis=0)
        pred_idx = int(np.argmax(mean_probs))

        explanation = self._build_explanation(
            loaded_model=loaded_model,
            segments=segments,
            centers=centers,
            mean_probs=mean_probs,
            segment_probs=segment_probs,
            fs=loaded.target_fs,
            detection_source=annotations.source,
        )

        segment_scores = segment_probs[:, pred_idx]
        segment_scores = segment_scores / (segment_scores.max() or 1.0)
        preview = downsample_for_preview(
            time_axis=np.arange(len(loaded.signal), dtype=np.float32) / float(loaded.target_fs),
            raw_signal=loaded.signal,
            filtered_signal=loaded.filtered_signal,
        )
        preview["highlight_regions"] = build_highlight_regions(
            centers=centers[: min(len(centers), len(segment_scores))],
            fs=loaded.target_fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            scores=segment_scores[: len(centers)],
        )

        beat_predictions = []
        for beat_index, (center, probs) in enumerate(zip(centers, segment_probs), start=1):
            beat_predictions.append(
                {
                    "beat_index": beat_index,
                    "sample": int(center),
                    "time_seconds": round(float(center) / float(loaded.target_fs), 4),
                    "class_name": CLASS_NAMES[int(np.argmax(probs))],
                    "confidence": round(float(np.max(probs)), 6),
                    "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))},
                }
            )

        return {
            "record": self._build_record_summary(
                record_name=loaded.record_name,
                original_fs=loaded.original_fs,
                target_fs=loaded.target_fs,
                signal=loaded.signal,
                analyzed_beats=len(segments),
                lead_name=loaded.lead_name,
            ),
            "prediction": {
                "predicted_class": pred_idx,
                "class_name": CLASS_NAMES[pred_idx],
                "confidence": round(float(mean_probs[pred_idx]), 6),
                "probabilities": {CLASS_NAMES[i]: round(float(mean_probs[i]), 6) for i in range(len(CLASS_NAMES))},
                "analyzed_segments": int(len(segments)),
                "lead_name": loaded.lead_name,
                "detection_source": annotations.source,
                "model_mode": loaded_model.mode,
                "model_version": loaded_model.metadata.get("model_version", "unknown"),
                "model_key": loaded_model.key,
                "model_label": loaded_model.label,
            },
            "explanation": explanation,
            "waveform_preview": preview,
            "beat_predictions": beat_predictions,
        }

    def get_pipeline_steps(
        self,
        record_path: Path,
        preferred_lead: Optional[str] = None,
        model_key: Optional[str] = None,
    ) -> Dict:
        loaded_model = self._get_model(model_key)
        analysis = self.analyze_record(record_path, preferred_lead=preferred_lead, model_key=loaded_model.key)
        record = analysis["record"]
        explanation = analysis["explanation"]

        steps = [
            {
                "step_name": "Lectura WFDB",
                "description": "Carga de `.dat + .hea`, seleccion automatica de derivacion y recuperacion de frecuencia original.",
                "details": {
                    "original_fs": record["original_fs"],
                    "target_fs": record["target_fs"],
                    "lead_name": record["lead_name"],
                },
            },
            {
                "step_name": "Preprocesado ECG",
                "description": "Remuestreo a frecuencia objetivo, filtro pasabanda y normalizacion por segmento.",
                "details": {
                    "window_size": self._window_size(loaded_model),
                    "pre_samples": self._pre_samples(loaded_model),
                    "post_samples": self._post_samples(loaded_model),
                },
            },
            {
                "step_name": "Deteccion y centrado",
                "description": "Uso de anotaciones clinicas `.atr` cuando existen o deteccion automatica de picos R cuando no estan disponibles.",
                "details": {
                    "detection_source": analysis["prediction"]["detection_source"],
                    "analyzed_segments": analysis["prediction"]["analyzed_segments"],
                },
            },
            {
                "step_name": "Inferencia recurrente",
                "description": f"Clasificacion por latido y agregacion de probabilidades mediante la arquitectura {loaded_model.label}.",
                "details": {
                    "predicted_class": analysis["prediction"]["class_name"],
                    "confidence": analysis["prediction"]["confidence"],
                },
            },
        ]

        return {
            "record": record,
            "preprocessing_steps": steps,
            "waveform_preview": analysis["waveform_preview"],
            "representative_segment": explanation["representative_segment"],
            "prediction_probabilities": analysis["prediction"]["probabilities"],
            "predicted_class": analysis["prediction"]["class_name"],
            "model_mode": loaded_model.mode,
            "model_key": loaded_model.key,
            "model_label": loaded_model.label,
        }

    def get_architecture_info(self, model_key: Optional[str] = None) -> Dict:
        loaded_model = self._get_model(model_key)
        if loaded_model.model is None:
            return {
                "model_key": loaded_model.key,
                "model_label": loaded_model.label,
                "model_mode": loaded_model.mode,
                "architecture_name": loaded_model.metadata.get("architecture_name"),
                "total_params": 0,
                "trainable_params": 0,
                "layers": [],
                "input_shape": None,
                "output_shape": None,
                "optimizer": None,
                "loss_function": None,
            }

        layers_info = []
        for layer in loaded_model.model.layers:
            layers_info.append(
                {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(layer.output_shape) if hasattr(layer, "output_shape") else None,
                    "num_params": int(layer.count_params()),
                    "trainable": bool(layer.trainable),
                }
            )

        optimizer = getattr(loaded_model.model, "optimizer", None)
        loss_function = getattr(loaded_model.model, "loss", None)

        return {
            "model_key": loaded_model.key,
            "model_label": loaded_model.label,
            "model_mode": loaded_model.mode,
            "architecture_name": loaded_model.metadata.get("architecture_name"),
            "total_params": int(loaded_model.model.count_params()),
            "trainable_params": int(sum(np.prod(weight.shape) for weight in loaded_model.model.trainable_weights)),
            "layers": layers_info,
            "input_shape": str(loaded_model.model.input_shape),
            "output_shape": str(loaded_model.model.output_shape),
            "optimizer": optimizer.__class__.__name__ if optimizer is not None else None,
            "loss_function": str(loss_function) if loss_function is not None else None,
        }

    def get_model_info(self, model_key: Optional[str] = None) -> Dict:
        loaded_model = self._get_model(model_key)
        return {
            "model_key": loaded_model.key,
            "model_label": loaded_model.label,
            "model_mode": loaded_model.mode,
            "metadata": loaded_model.metadata,
            "available_models": self.available_models(),
        }


service = ModelService()
