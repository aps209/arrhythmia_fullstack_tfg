"""
model_service.py — Servicio central de inferencia y explicabilidad.

Carga el modelo Keras entrenado (o utiliza heurísticas en modo demo),
y proporciona:
  - Predicción de arritmias a partir de 15 intervalos R-R
  - Explicabilidad por oclusión (timestep importance)
  - Información de la arquitectura del modelo
  - Pasos intermedios del pipeline para visualización educativa
"""

import json
import logging
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from .config import MODEL_PATH, SCALER_PATH, METADATA_PATH, CLASS_NAMES, N_STEPS

logger = logging.getLogger(__name__)


class ModelService:
    """
    Servicio singleton que encapsula toda la lógica de predicción,
    explicabilidad y visualización del pipeline de la red neuronal.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata: Dict = {
            "model_version": "demo-0.1.0",
            "input_type": "15_rr_intervals",
            "feature_channels": ["rr", "rr_diff", "rolling_mean", "rr_zscore"],
            "split_strategy": "group_split_placeholder",
        }
        self.mode: str = "demo"
        self._load()

    # ─── Carga de artefactos ───────────────────────────────────────────

    def _load(self) -> None:
        """Carga modelo, scaler y metadata desde disco si existen."""
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                self.model = load_model(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.mode = "trained"
                logger.info("Modelo cargado correctamente desde %s", MODEL_PATH)
            except Exception as e:
                logger.error("Error al cargar el modelo: %s", e)

        if METADATA_PATH.exists():
            try:
                self.metadata.update(
                    json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                )
                logger.info("Metadata cargada desde %s", METADATA_PATH)
            except Exception as e:
                logger.error("Error al cargar metadata: %s", e)

    # ─── Derivación de features ────────────────────────────────────────

    def _derive_features(self, rr: np.ndarray) -> np.ndarray:
        """
        Deriva 4 canales de features a partir de los R-R crudos:
          1. rr: intervalos originales
          2. rr_diff: diferencia entre intervalos consecutivos
          3. rolling_mean: media móvil de ventana 3
          4. rr_zscore: z-score normalizado
        """
        rr = rr.astype(float)
        rr_diff = np.diff(rr, prepend=rr[0])
        rolling = np.array([rr[max(0, i - 2):i + 1].mean() for i in range(len(rr))])
        rr_std = rr.std() if rr.std() > 1e-8 else 1.0
        rr_z = (rr - rr.mean()) / rr_std
        feats = np.stack([rr, rr_diff, rolling, rr_z], axis=-1)
        return feats

    # ─── Estadísticas de resumen ───────────────────────────────────────

    def _summarize(self, rr: np.ndarray) -> Dict[str, float]:
        """Calcula estadísticas descriptivas de la secuencia R-R."""
        return {
            "rr_mean": float(rr.mean()),
            "rr_std": float(rr.std()),
            "rr_min": float(rr.min()),
            "rr_max": float(rr.max()),
            "rr_range": float(rr.max() - rr.min()),
        }

    # ─── Predicción heurística (modo demo) ─────────────────────────────

    def _heuristic_predict(self, rr: np.ndarray) -> np.ndarray:
        """
        Predicción basada en heurísticas cuando no hay modelo entrenado.
        Simula probabilidades en base a estadísticas de la señal R-R.
        """
        rr_std = float(rr.std())
        rr_range = float(rr.max() - rr.min())
        rr_diff_mean = float(np.abs(np.diff(rr)).mean())

        base = np.array([0.70, 0.08, 0.12, 0.05, 0.05], dtype=float)

        if rr_std > 0.14 or rr_range > 0.35:
            base += np.array([-0.25, 0.10, 0.18, 0.02, -0.05])

        if rr_diff_mean > 0.10:
            base += np.array([-0.10, 0.10, 0.06, 0.01, -0.07])

        if rr.mean() < 0.55:
            base += np.array([-0.05, 0.03, 0.12, 0.00, -0.10])

        base = np.clip(base, 1e-6, None)
        base /= base.sum()
        return base

    # ─── Predicción principal ──────────────────────────────────────────

    def predict_proba(self, rr_intervals: List[float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predice probabilidades de cada clase de arritmia.

        Returns
        -------
        tuple[np.ndarray, dict]
            Array de probabilidades y diccionario de estadísticas de resumen.
        """
        rr = np.array(rr_intervals, dtype=float)
        summary = self._summarize(rr)

        if self.mode == "trained":
            feats = self._derive_features(rr)
            flat = feats.reshape(1, -1)
            scaled = self.scaler.transform(flat).reshape(1, N_STEPS, -1)
            probs = self.model.predict(scaled, verbose=0)[0]
            return probs, summary

        return self._heuristic_predict(rr), summary

    # ─── Explicabilidad por oclusión ───────────────────────────────────

    def explain(self, rr_intervals: List[float]) -> Dict:
        """
        Genera explicación de la predicción usando oclusión de timesteps.

        Cada timestep se reemplaza por la media y se mide la caída en
        probabilidad de la clase predicha, identificando los pasos más
        influyentes en la decisión.
        """
        rr = np.array(rr_intervals, dtype=float)
        probs, _ = self.predict_proba(rr_intervals)
        pred_idx = int(np.argmax(probs))
        baseline = float(probs[pred_idx])

        importances = []
        replacement = float(rr.mean())

        for i in range(len(rr)):
            occluded = rr.copy()
            occluded[i] = replacement
            occluded_probs, _ = self.predict_proba(occluded.tolist())
            delta = max(0.0, baseline - float(occluded_probs[pred_idx]))
            importances.append(delta)

        total = sum(importances) or 1.0
        importances = [float(v / total) for v in importances]
        top = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:3]

        clinical_text = (
            f"La predicción se apoya sobre todo en los pasos {', '.join(str(i + 1) for i in top)} "
            f"de la ventana R-R. Cambios bruscos o variabilidad elevada cerca de esos puntos "
            f"aumentan el riesgo estimado de {CLASS_NAMES[pred_idx]}."
        )

        return {
            "predicted_class": pred_idx,
            "class_name": CLASS_NAMES[pred_idx],
            "confidence": float(baseline),
            "timestep_importance": importances,
            "top_timesteps": [i + 1 for i in top],
            "clinical_text": clinical_text,
            "model_mode": self.mode,
            "model_version": self.metadata.get("model_version", "unknown"),
        }

    # ─── Respuesta de predicción formateada ────────────────────────────

    def make_prediction_response(self, rr_intervals: List[float]) -> Dict:
        """
        Genera respuesta completa de predicción con clase, confianza,
        nivel de riesgo y distribución de probabilidades.
        """
        probs, summary = self.predict_proba(rr_intervals)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        risk_level = (
            "high" if pred_idx != 0 and confidence >= 0.6
            else "medium" if pred_idx != 0
            else "low"
        )

        return {
            "predicted_class": pred_idx,
            "class_name": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "risk_level": risk_level,
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
            "input_summary": summary,
            "model_mode": self.mode,
            "model_version": self.metadata.get("model_version", "unknown"),
        }

    # ─── Arquitectura del modelo ───────────────────────────────────────

    def get_architecture_info(self) -> Dict:
        """
        Devuelve información detallada de la arquitectura del modelo:
        capas, parámetros, shapes y configuración del optimizador.
        """
        if self.mode == "trained" and self.model is not None:
            layers_info = []
            for layer in self.model.layers:
                layer_config = {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(layer.output_shape) if hasattr(layer, "output_shape") else None,
                    "num_params": int(layer.count_params()),
                    "trainable": layer.trainable,
                }
                layers_info.append(layer_config)

            return {
                "model_mode": self.mode,
                "total_params": int(self.model.count_params()),
                "trainable_params": int(sum(
                    np.prod(w.shape) for w in self.model.trainable_weights
                )),
                "layers": layers_info,
                "input_shape": str(self.model.input_shape),
                "output_shape": str(self.model.output_shape),
                "optimizer": self.model.optimizer.__class__.__name__ if self.model.optimizer else None,
                "loss_function": str(self.model.loss) if hasattr(self.model, "loss") else None,
            }

        # Modo demo: devolver la arquitectura esperada
        return {
            "model_mode": self.mode,
            "total_params": 82213,
            "trainable_params": 81829,
            "layers": [
                {"name": "input", "type": "InputLayer", "output_shape": "(None, 15, 4)", "num_params": 0, "trainable": False},
                {"name": "conv1d", "type": "Conv1D", "output_shape": "(None, 15, 32)", "num_params": 416, "trainable": True},
                {"name": "batch_norm", "type": "BatchNormalization", "output_shape": "(None, 15, 32)", "num_params": 128, "trainable": True},
                {"name": "bidirectional_lstm_1", "type": "Bidirectional(LSTM)", "output_shape": "(None, 15, 128)", "num_params": 49664, "trainable": True},
                {"name": "dropout_1", "type": "Dropout", "output_shape": "(None, 15, 128)", "num_params": 0, "trainable": False},
                {"name": "bidirectional_lstm_2", "type": "Bidirectional(LSTM)", "output_shape": "(None, 64)", "num_params": 41216, "trainable": True},
                {"name": "dropout_2", "type": "Dropout", "output_shape": "(None, 64)", "num_params": 0, "trainable": False},
                {"name": "dense_hidden", "type": "Dense(relu)", "output_shape": "(None, 64)", "num_params": 4160, "trainable": True},
                {"name": "dense_output", "type": "Dense(softmax)", "output_shape": "(None, 5)", "num_params": 325, "trainable": True},
            ],
            "input_shape": "(None, 15, 4)",
            "output_shape": "(None, 5)",
            "optimizer": "Adam",
            "loss_function": "sparse_categorical_crossentropy",
        }

    # ─── Pasos intermedios del pipeline ────────────────────────────────

    def get_pipeline_steps(self, rr_intervals: List[float]) -> Dict:
        """
        Devuelve los pasos intermedios del pipeline de predicción
        para visualización educativa:
          1. R-R crudos
          2. Features derivadas (rr_diff, rolling_mean, rr_zscore)
          3. Valores normalizados (si scaler disponible)
          4. Predicción final
        """
        rr = np.array(rr_intervals, dtype=float)

        # Paso 1: Features derivadas
        feats = self._derive_features(rr)
        rr_vals = feats[:, 0].tolist()
        rr_diff = feats[:, 1].tolist()
        rolling = feats[:, 2].tolist()
        rr_z = feats[:, 3].tolist()

        derived_features = [
            {
                "step_name": "R-R Intervals",
                "description": "Intervalos R-R originales en segundos",
                "values": [round(v, 6) for v in rr_vals],
            },
            {
                "step_name": "ΔR-R (Diferencia)",
                "description": "Diferencia entre intervalos consecutivos: rr[i] - rr[i-1]",
                "values": [round(v, 6) for v in rr_diff],
            },
            {
                "step_name": "Media Móvil (ventana=3)",
                "description": "Promedio de los últimos 3 intervalos para suavizar la señal",
                "values": [round(v, 6) for v in rolling],
            },
            {
                "step_name": "Z-Score",
                "description": "Normalización estadística: (rr - μ) / σ",
                "values": [round(v, 6) for v in rr_z],
            },
        ]

        # Paso 2: Normalización con scaler
        normalized = None
        if self.scaler is not None:
            flat = feats.reshape(1, -1)
            scaled = self.scaler.transform(flat).reshape(N_STEPS, -1)
            normalized = [[round(float(v), 6) for v in row] for row in scaled]

        # Paso 3: Predicción
        probs, _ = self.predict_proba(rr_intervals)
        pred_idx = int(np.argmax(probs))

        return {
            "raw_rr": rr_intervals,
            "derived_features": derived_features,
            "normalized_sample": normalized,
            "prediction_probabilities": {
                CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(probs))
            },
            "predicted_class": CLASS_NAMES[pred_idx],
            "model_mode": self.mode,
        }


# Instancia singleton
service = ModelService()
