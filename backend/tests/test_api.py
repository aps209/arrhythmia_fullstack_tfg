"""
test_api.py — Tests de integración para la API de Arrhythmia Early Warning.

Valida todos los endpoints con secuencias válidas e inválidas,
incluyendo predicción, explicabilidad, ECG y pipeline.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ─── Datos de prueba ──────────────────────────────────────────────────

VALID_RR = [0.82, 0.79, 0.81, 0.80, 0.78, 0.77, 0.83, 0.84,
            0.82, 0.80, 0.81, 0.79, 0.78, 0.80, 0.81]

INVALID_RR_SHORT = [0.82, 0.79, 0.81]  # Menos de 15

INVALID_RR_RANGE = [0.82, 0.79, 0.81, 0.80, 0.78, 0.77, 0.83, 0.84,
                     0.82, 0.80, 0.81, 0.79, -0.5, 0.80, 0.81]  # Valor negativo


# ─── Tests de estado ──────────────────────────────────────────────────


def test_health():
    """Verifica que el endpoint de health responde correctamente."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_mode" in data
    assert "model_version" in data


def test_model_info():
    """Verifica que /model-info devuelve metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_mode" in data
    assert "metadata" in data


def test_model_architecture():
    """Verifica que /model-architecture devuelve info de capas."""
    response = client.get("/model-architecture")
    assert response.status_code == 200
    data = response.json()
    assert "model_mode" in data
    assert "total_params" in data
    assert "layers" in data
    assert len(data["layers"]) > 0
    assert "input_shape" in data
    assert "output_shape" in data


# ─── Tests de predicción ──────────────────────────────────────────────


def test_predict():
    """Verifica predicción con secuencia válida."""
    response = client.post("/predict", json={"rr_intervals": VALID_RR})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "class_name" in data
    assert "confidence" in data
    assert "risk_level" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 5
    assert 0 <= data["confidence"] <= 1


def test_explain():
    """Verifica explicabilidad con secuencia válida."""
    response = client.post("/explain", json={"rr_intervals": VALID_RR})
    assert response.status_code == 200
    data = response.json()
    assert "timestep_importance" in data
    assert len(data["timestep_importance"]) == 15
    assert "top_timesteps" in data
    assert "clinical_text" in data


def test_predict_and_explain():
    """Verifica predicción + explicabilidad combinadas."""
    response = client.post("/predict-and-explain", json={"rr_intervals": VALID_RR})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "explanation" in data
    assert data["prediction"]["class_name"] == data["explanation"]["class_name"]


# ─── Tests de visualización ───────────────────────────────────────────


def test_ecg_signal():
    """Verifica generación de señal ECG sintética."""
    response = client.post("/ecg-signal", json={"rr_intervals": VALID_RR})
    assert response.status_code == 200
    data = response.json()
    assert "ecg_image_base64" in data
    assert len(data["ecg_image_base64"]) > 100  # Base64 debería ser largo
    assert "duration_seconds" in data
    assert "num_beats" in data
    assert data["num_beats"] == 15
    assert "mean_heart_rate_bpm" in data
    assert data["mean_heart_rate_bpm"] > 0


def test_pipeline_steps():
    """Verifica pasos intermedios del pipeline."""
    response = client.post("/pipeline-steps", json={"rr_intervals": VALID_RR})
    assert response.status_code == 200
    data = response.json()
    assert "raw_rr" in data
    assert len(data["raw_rr"]) == 15
    assert "derived_features" in data
    assert len(data["derived_features"]) == 4  # 4 canales
    assert "prediction_probabilities" in data
    assert "predicted_class" in data


# ─── Tests de validación ──────────────────────────────────────────────


def test_predict_invalid_short():
    """Verifica que secuencias cortas son rechazadas."""
    response = client.post("/predict", json={"rr_intervals": INVALID_RR_SHORT})
    assert response.status_code == 422  # Validation error


def test_predict_invalid_range():
    """Verifica que valores fuera de rango son rechazados."""
    response = client.post("/predict", json={"rr_intervals": INVALID_RR_RANGE})
    assert response.status_code == 422


def test_predict_empty():
    """Verifica que secuencia vacía es rechazada."""
    response = client.post("/predict", json={"rr_intervals": []})
    assert response.status_code == 422


def test_predict_no_body():
    """Verifica que petición sin body es rechazada."""
    response = client.post("/predict", json={})
    assert response.status_code == 422
