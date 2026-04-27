from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def wfdb_uploads():
    return [
        ("files", ("100.hea", b"fake header", "text/plain")),
        ("files", ("100.dat", b"fake data", "application/octet-stream")),
        ("files", ("100.atr", b"fake atr", "application/octet-stream")),
    ]


def fake_analysis():
    return {
        "record": {
            "record_name": "100",
            "original_fs": 360.0,
            "target_fs": 250.0,
            "duration_seconds": 10.0,
            "num_samples": 2500,
            "analyzed_beats": 3,
            "lead_name": "MLII",
        },
        "prediction": {
            "predicted_class": 2,
            "class_name": "VEB",
            "confidence": 0.81,
            "probabilities": {"Normal": 0.05, "SVEB": 0.06, "VEB": 0.81, "Fusion": 0.05, "Unknown": 0.03},
            "analyzed_segments": 3,
            "lead_name": "MLII",
            "detection_source": "annotations",
            "model_mode": "trained",
            "model_version": "test-model",
            "model_key": "conv1d_bilstm",
            "model_label": "Conv1D + BiLSTM",
        },
        "explanation": {
            "predicted_class": 2,
            "class_name": "VEB",
            "confidence": 0.81,
            "technical_summary": "Resumen técnico.",
            "evidence": ["e1", "e2"],
            "limitations": ["l1"],
            "detection_source": "annotations",
            "top_beats": [1, 2, 3],
            "representative_segment": {
                "beat_index": 1,
                "signal": [0.1, 0.2, 0.1],
                "saliency": [0.0, 1.0, 0.2],
                "salient_region_ms": {"start_ms": 10.0, "end_ms": 30.0, "width_ms": 20.0},
            },
            "model_mode": "trained",
            "model_version": "test-model",
            "model_key": "conv1d_bilstm",
            "model_label": "Conv1D + BiLSTM",
        },
        "waveform_preview": {
            "time": [0.0, 0.1, 0.2],
            "raw_signal": [0.0, 0.1, 0.0],
            "filtered_signal": [0.0, 0.08, 0.0],
            "highlight_regions": [
                {"beat_index": 1, "start_time": 0.0, "end_time": 0.2, "score": 0.9}
            ],
        },
        "beat_predictions": [
            {
                "beat_index": 1,
                "sample": 50,
                "time_seconds": 0.2,
                "class_name": "VEB",
                "confidence": 0.88,
                "probabilities": {"Normal": 0.02, "SVEB": 0.05, "VEB": 0.88, "Fusion": 0.03, "Unknown": 0.02},
            }
        ],
    }


def fake_pipeline():
    return {
        "record": {
            "record_name": "100",
            "original_fs": 360.0,
            "target_fs": 250.0,
            "duration_seconds": 10.0,
            "num_samples": 2500,
            "analyzed_beats": 3,
            "lead_name": "MLII",
        },
        "preprocessing_steps": [
            {
                "step_name": "Lectura WFDB",
                "description": "desc",
                "details": {"original_fs": 360.0, "target_fs": 250.0, "lead_name": "MLII"},
            }
        ],
        "waveform_preview": {
            "time": [0.0, 0.1],
            "raw_signal": [0.0, 0.1],
            "filtered_signal": [0.0, 0.08],
            "highlight_regions": [],
        },
        "representative_segment": {
            "beat_index": 1,
            "signal": [0.1, 0.2],
            "saliency": [0.3, 1.0],
            "salient_region_ms": {"start_ms": 10.0, "end_ms": 20.0, "width_ms": 10.0},
        },
        "prediction_probabilities": {"Normal": 0.1, "SVEB": 0.1, "VEB": 0.7, "Fusion": 0.05, "Unknown": 0.05},
        "predicted_class": "VEB",
        "model_mode": "trained",
        "model_key": "conv1d_bilstm",
        "model_label": "Conv1D + BiLSTM",
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["expected_input"] == "wfdb_bundle_dat_hea_optional_atr"
    assert "default_model_key" in data
    assert "available_models" in data


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_mode" in data
    assert "metadata" in data


def test_model_architecture():
    response = client.get("/model-architecture")
    assert response.status_code == 200
    data = response.json()
    assert "model_mode" in data
    assert "layers" in data


def test_analyze_record(monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.service, "analyze_record", lambda *args, **kwargs: fake_analysis())
    response = client.post("/analyze-record", files=wfdb_uploads(), data={"model_key": "conv1d_bilstm"})

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]["class_name"] == "VEB"
    assert data["record"]["lead_name"] == "MLII"
    assert len(data["beat_predictions"]) == 1
    assert data["prediction"]["model_key"] == "conv1d_bilstm"


def test_predict_and_explain_alias(monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.service, "analyze_record", lambda *args, **kwargs: fake_analysis())
    response = client.post("/predict-and-explain", files=wfdb_uploads(), data={"model_key": "conv1d_bilstm"})

    assert response.status_code == 200
    assert response.json()["explanation"]["class_name"] == "VEB"


def test_pipeline_steps(monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.service, "get_pipeline_steps", lambda *args, **kwargs: fake_pipeline())
    response = client.post("/pipeline-steps", files=wfdb_uploads(), data={"model_key": "conv1d_bilstm"})

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "VEB"
    assert len(data["preprocessing_steps"]) == 1
    assert data["model_key"] == "conv1d_bilstm"


def test_invalid_uploads():
    response = client.post(
        "/analyze-record",
        files=[("files", ("100.dat", b"fake data", "application/octet-stream"))],
    )
    assert response.status_code == 422
