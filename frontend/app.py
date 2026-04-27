from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8501"))

APP_NAME = "Arrhythmia Review"
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "toggleSpikelines",
    ],
}
NAV_ITEMS = [
    {"path": "/", "label": "Resumen"},
    {"path": "/prediction", "label": "Prediccion"},
    {"path": "/pipeline", "label": "Pipeline"},
    {"path": "/system", "label": "Sistema"},
]
CLASS_EXPLANATIONS = {
    "Normal": "El patron agregado es compatible con actividad sinusal estable dentro del conjunto de latidos revisados.",
    "SVEB": "La salida sugiere un predominio de latidos supraventriculares prematuros en el trazado analizado.",
    "VEB": "La evidencia del modelo se concentra en morfologias compatibles con ectopia ventricular.",
    "Fusion": "La clasificacion indica mezcla de rasgos normales y ventriculares en los latidos mas influyentes.",
    "Unknown": "La distribucion de probabilidad no es suficientemente estable para asignar una superclase AAMI con claridad.",
}


app = FastAPI(title=APP_NAME, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def safe_text(value: Any, default: str = "No disponible") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def api_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("message") or payload)
    return str(payload)


def request_backend_json(endpoint: str, timeout: int = 8, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=timeout, params=params)
    if response.status_code != 200:
        raise RuntimeError(api_error_message(response))
    return response.json()


def forward_files_to_backend(
    endpoint: str,
    files: list[UploadFile],
    form_data: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    request_files: list[tuple[str, tuple[str, bytes, str]]] = []
    for upload in files:
        filename = upload.filename or ""
        if not filename:
            continue
        payload = upload.file.read()
        request_files.append(("files", (filename, payload, "application/octet-stream")))

    response = requests.post(
        f"{BACKEND_URL}{endpoint}",
        files=request_files,
        data=form_data or {},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(api_error_message(response))
    return response.json()


def has_wfdb_bundle(files: list[UploadFile]) -> bool:
    stems: dict[str, set[str]] = {}
    for upload in files:
        if not upload.filename:
            continue
        path = Path(upload.filename)
        stems.setdefault(path.stem, set()).add(path.suffix.lower())
    return any(".dat" in suffixes and ".hea" in suffixes for suffixes in stems.values())


def render_plot(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)


def dataframe_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="table-empty">No hay datos disponibles para mostrar.</div>'
    return df.to_html(index=False, border=0, classes="data-table")


def build_status_snapshot() -> dict[str, Any]:
    try:
        health = request_backend_json("/health", timeout=4)
        connected = True
    except Exception as exc:
        health = {"status": "error", "detail": safe_text(exc)}
        connected = False

    return {
        "connected": connected,
        "backend_label": "Disponible" if connected else "No disponible",
        "backend_detail": safe_text(health.get("detail"), "Servicio operativo" if connected else "Sin respuesta"),
        "model_mode": safe_text(health.get("model_mode"), "unknown"),
        "model_version": safe_text(health.get("model_version"), "unknown"),
        "status": safe_text(health.get("status"), "unknown"),
        "default_model_key": safe_text(health.get("default_model_key"), "conv1d_bilstm"),
        "available_models": health.get("available_models", []),
        "selected_model": health.get("selected_model", {}),
    }


def build_navigation(current_path: str) -> list[dict[str, Any]]:
    return [
        {
            "path": item["path"],
            "label": item["label"],
            "active": current_path == item["path"],
        }
        for item in NAV_ITEMS
    ]


def normalize_model_key(requested_key: str | None, catalog: list[dict[str, Any]], fallback: str) -> str:
    catalog_keys = {safe_text(item.get("model_key"), "") for item in catalog}
    if requested_key and requested_key in catalog_keys:
        return requested_key
    return fallback


def model_option_label(model: dict[str, Any]) -> str:
    label = safe_text(model.get("model_label"))
    mode = safe_text(model.get("model_mode")).lower()
    if mode == "trained":
        return f"{label} - disponible"
    if mode == "unavailable":
        return f"{label} - pendiente de entrenamiento"
    return f"{label} - {mode}"


def find_model(model_key: str, catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    for model in catalog:
        if safe_text(model.get("model_key"), "") == model_key:
            return model
    return None


def base_context(
    request: Request,
    page_title: str,
    page_description: str,
    selected_model_key: str | None = None,
) -> dict[str, Any]:
    status = build_status_snapshot()
    model_catalog = [
        {**model, "option_label": model_option_label(model)}
        for model in status.get("available_models", [])
    ]
    normalized_model_key = normalize_model_key(
        selected_model_key,
        model_catalog,
        status.get("default_model_key", "conv1d_bilstm"),
    )
    return {
        "request": request,
        "app_name": APP_NAME,
        "page_title": page_title,
        "page_description": page_description,
        "title": f"{page_title} | {APP_NAME}",
        "nav_items": build_navigation(request.url.path),
        "status_snapshot": status,
        "include_plotly": False,
        "error_message": None,
        "success_message": None,
        "model_catalog": model_catalog,
        "selected_model_key": normalized_model_key,
        "selected_model": find_model(normalized_model_key, model_catalog),
        "default_model_label": safe_text(status.get("selected_model", {}).get("model_label")),
    }


def workflow_steps() -> list[str]:
    return [
        "Cargar un bundle WFDB con el par .dat + .hea y, cuando exista, el archivo .atr asociado al mismo estudio.",
        "Seleccionar el modelo recurrente a comparar y enviar el registro para revision automatizada.",
        "Revisar la prediccion agregada, la distribucion de probabilidad y la explicacion tecnica del resultado.",
        "Consultar el pipeline para inspeccionar preprocesado, arquitectura y segmento representativo del analisis.",
    ]


def architecture_summary(model_key: str | None = None) -> dict[str, Any] | None:
    try:
        params = {"model_key": model_key} if model_key else None
        return request_backend_json("/model-architecture", timeout=5, params=params)
    except Exception:
        return None


def metadata_summary(model_key: str | None = None) -> dict[str, Any] | None:
    try:
        params = {"model_key": model_key} if model_key else None
        return request_backend_json("/model-info", timeout=5, params=params)
    except Exception:
        return None


def prediction_summary_text(prediction: dict[str, Any]) -> str:
    class_name = safe_text(prediction.get("class_name"))
    confidence = float(prediction.get("confidence", 0.0))
    base = CLASS_EXPLANATIONS.get(class_name, "El patron dominante requiere revision adicional por parte del profesional.")
    return (
        f"El estudio se resume como {class_name} con una confianza del {confidence:.1%}. "
        f"{base} Esta salida se presenta como apoyo tecnico a la revision clinica y no sustituye la valoracion profesional."
    )


def waveform_figure(preview: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preview["time"],
            y=preview["raw_signal"],
            name="Senal original",
            mode="lines",
            line=dict(color="#7b8480", width=1),
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=preview["time"],
            y=preview["filtered_signal"],
            name="Senal filtrada",
            mode="lines",
            line=dict(color="#21584f", width=1.8),
        )
    )

    for region in preview.get("highlight_regions", [])[:20]:
        fig.add_vrect(
            x0=region["start_time"],
            x1=region["end_time"],
            fillcolor="#d9e7e3",
            opacity=min(0.35, 0.08 + region["score"] * 0.25),
            line_width=0,
        )

    fig.update_layout(
        title="Senal ECG del registro",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#182120", family="IBM Plex Sans"),
        height=400,
        margin=dict(l=12, r=12, t=44, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Tiempo (s)", gridcolor="#e0e6e3", color="#596360"),
        yaxis=dict(title="Amplitud", gridcolor="#e0e6e3", color="#596360"),
    )
    return fig


def probabilities_figure(probabilities: dict[str, float], selected_class: str, title: str) -> go.Figure:
    rows = sorted(probabilities.items(), key=lambda item: item[1])
    fig = go.Figure(
        go.Bar(
            x=[value for _, value in rows],
            y=[name for name, _ in rows],
            orientation="h",
            marker=dict(
                color=[
                    "#21584f" if name == selected_class else "#c5ceca"
                    for name, _ in rows
                ]
            ),
            text=[f"{value:.2%}" for _, value in rows],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#182120", family="IBM Plex Sans"),
        height=320,
        margin=dict(l=12, r=36, t=42, b=12),
        xaxis=dict(range=[0, 1.05], gridcolor="#e0e6e3", color="#596360"),
        yaxis=dict(color="#596360"),
    )
    return fig


def preview_figure(preview: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preview["time"],
            y=preview["raw_signal"],
            mode="lines",
            name="Original",
            line=dict(color="#7b8480", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=preview["time"],
            y=preview["filtered_signal"],
            mode="lines",
            name="Filtrada",
            line=dict(color="#21584f", width=1.8),
        )
    )
    fig.update_layout(
        title="Previsualizacion de preprocesado",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#182120", family="IBM Plex Sans"),
        height=360,
        margin=dict(l=12, r=12, t=42, b=12),
        xaxis=dict(title="Tiempo (s)", gridcolor="#e0e6e3", color="#596360"),
        yaxis=dict(title="Amplitud", gridcolor="#e0e6e3", color="#596360"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def representative_segment_figure(segment: dict[str, Any]) -> go.Figure:
    x_values = list(range(len(segment["signal"])))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=segment["signal"],
            mode="lines",
            name="ECG",
            line=dict(color="#21584f", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=segment["saliency"],
            mode="lines",
            name="Sensibilidad",
            line=dict(color="#7b4d3a", width=1.8),
            fill="tozeroy",
            opacity=0.35,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Latido representativo {segment['beat_index']}",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#182120", family="IBM Plex Sans"),
        height=340,
        margin=dict(l=12, r=12, t=42, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title="Muestras", gridcolor="#e0e6e3", color="#596360")
    fig.update_yaxes(title="ECG", gridcolor="#e0e6e3", color="#596360", secondary_y=False)
    fig.update_yaxes(title="Sensibilidad", gridcolor="#e0e6e3", color="#596360", secondary_y=True)
    return fig


def build_prediction_context(payload: dict[str, Any]) -> dict[str, Any]:
    record = payload["record"]
    prediction = payload["prediction"]
    explanation = payload["explanation"]

    return {
        "has_result": True,
        "prediction": prediction,
        "record": record,
        "explanation": explanation,
        "prediction_summary": prediction_summary_text(prediction),
        "waveform_plot": render_plot(waveform_figure(payload["waveform_preview"])),
        "probabilities_plot": render_plot(
            probabilities_figure(
                prediction["probabilities"],
                prediction["class_name"],
                "Distribucion de probabilidad por clase",
            )
        ),
        "selected_model_key": prediction.get("model_key"),
    }


def build_pipeline_context(payload: dict[str, Any], arch: dict[str, Any] | None) -> dict[str, Any]:
    steps = pd.DataFrame(
        [
            {
                "Paso": item["step_name"],
                "Descripcion": item["description"],
                "Detalles": ", ".join(f"{key}={value}" for key, value in item["details"].items()),
            }
            for item in payload["preprocessing_steps"]
        ]
    )

    layers = pd.DataFrame(arch.get("layers", [])) if arch else pd.DataFrame()
    result: dict[str, Any] = {
        "has_result": True,
        "pipeline_plot": render_plot(preview_figure(payload["waveform_preview"])),
        "steps_table_html": dataframe_to_html(steps),
        "layers_table_html": dataframe_to_html(layers),
        "pipeline_probabilities_plot": render_plot(
            probabilities_figure(
                payload["prediction_probabilities"],
                payload["predicted_class"],
                f"Clase agregada: {payload['predicted_class']}",
            )
        ),
        "pipeline_record": payload["record"],
        "pipeline_architecture": arch,
        "representative_segment": payload.get("representative_segment"),
        "pipeline_model_label": payload.get("model_label"),
        "selected_model_key": payload.get("model_key"),
    }

    representative = payload.get("representative_segment")
    if representative:
        result["representative_plot"] = render_plot(representative_segment_figure(representative))

    return result


def metadata_table(model_info: dict[str, Any] | None) -> str:
    if not model_info:
        return '<div class="table-empty">No ha sido posible recuperar la metadata del modelo.</div>'

    metadata = model_info.get("metadata", {})
    rows = [
        {"Clave": key, "Valor": str(value)}
        for key, value in metadata.items()
        if key not in {"history", "report", "confusion_matrix"}
    ]
    return dataframe_to_html(pd.DataFrame(rows))


def model_catalog_table(model_catalog: list[dict[str, Any]]) -> str:
    rows = []
    for item in model_catalog:
        rows.append(
            {
                "Modelo": safe_text(item.get("model_label")),
                "Arquitectura": safe_text(item.get("architecture_name")),
                "Estado": "Disponible" if safe_text(item.get("model_mode")).lower() == "trained" else "Pendiente",
                "Version": safe_text(item.get("model_version")),
                "Frecuencia": f"{safe_text(item.get('target_fs'))} Hz",
                "Ventana": safe_text(item.get("window_size")),
            }
        )
    return dataframe_to_html(pd.DataFrame(rows))


@app.get("/", response_class=HTMLResponse)
def overview(request: Request) -> HTMLResponse:
    context = base_context(
        request,
        "Resumen del servicio",
        "Vista operativa para revisar disponibilidad del sistema, catalogo de modelos y flujo de trabajo.",
    )
    context["architecture"] = architecture_summary(context["selected_model_key"])
    context["workflow_steps"] = workflow_steps()
    context["model_catalog_table_html"] = model_catalog_table(context["model_catalog"])
    return templates.TemplateResponse("overview.html", context)


@app.get("/prediction", response_class=HTMLResponse)
def prediction_page(request: Request) -> HTMLResponse:
    context = base_context(
        request,
        "Prediccion clinica",
        "Carga del estudio WFDB y revision del resultado agregado con la misma salida para cada modelo recurrente disponible.",
    )
    context["has_result"] = False
    context["include_plotly"] = True
    return templates.TemplateResponse("prediction.html", context)


@app.post("/prediction", response_class=HTMLResponse)
def prediction_submit(
    request: Request,
    files: list[UploadFile] | None = File(default=None),
    model_key: str = Form(default=""),
) -> HTMLResponse:
    context = base_context(
        request,
        "Prediccion clinica",
        "Carga del estudio WFDB y revision del resultado agregado con la misma salida para cada modelo recurrente disponible.",
        selected_model_key=model_key or None,
    )
    context["include_plotly"] = True
    context["has_result"] = False
    uploads = files or []

    if not uploads:
        context["error_message"] = "Debes subir al menos los ficheros .dat y .hea del mismo registro."
        return templates.TemplateResponse("prediction.html", context)

    if not has_wfdb_bundle(uploads):
        context["error_message"] = "No se ha detectado un par .dat + .hea valido del mismo registro."
        return templates.TemplateResponse("prediction.html", context)

    form_data = {"model_key": context["selected_model_key"]}

    try:
        payload = forward_files_to_backend("/analyze-record", uploads, form_data=form_data, timeout=120)
        context.update(build_prediction_context(payload))
        context["selected_model"] = find_model(context["selected_model_key"], context["model_catalog"])
        context["success_message"] = "Prediccion completada correctamente."
    except Exception as exc:
        context["error_message"] = safe_text(exc)

    return templates.TemplateResponse("prediction.html", context)


@app.get("/pipeline", response_class=HTMLResponse)
def pipeline_page(request: Request) -> HTMLResponse:
    context = base_context(
        request,
        "Pipeline de analisis",
        "Revision tecnica del preprocesado, arquitectura y salida probabilistica para el modelo seleccionado.",
    )
    context["include_plotly"] = True
    context["has_result"] = False
    context["architecture"] = architecture_summary(context["selected_model_key"])
    return templates.TemplateResponse("pipeline.html", context)


@app.post("/pipeline", response_class=HTMLResponse)
def pipeline_submit(
    request: Request,
    files: list[UploadFile] | None = File(default=None),
    model_key: str = Form(default=""),
) -> HTMLResponse:
    context = base_context(
        request,
        "Pipeline de analisis",
        "Revision tecnica del preprocesado, arquitectura y salida probabilistica para el modelo seleccionado.",
        selected_model_key=model_key or None,
    )
    context["include_plotly"] = True
    context["has_result"] = False
    uploads = files or []

    if not uploads:
        context["error_message"] = "Debes subir un bundle WFDB para revisar el pipeline."
        context["architecture"] = architecture_summary(context["selected_model_key"])
        return templates.TemplateResponse("pipeline.html", context)

    if not has_wfdb_bundle(uploads):
        context["error_message"] = "No se ha detectado un par .dat + .hea valido del mismo registro."
        context["architecture"] = architecture_summary(context["selected_model_key"])
        return templates.TemplateResponse("pipeline.html", context)

    form_data = {"model_key": context["selected_model_key"]}

    try:
        payload = forward_files_to_backend("/pipeline-steps", uploads, form_data=form_data, timeout=120)
        arch = architecture_summary(context["selected_model_key"])
        context.update(build_pipeline_context(payload, arch))
        context["architecture"] = arch
        context["selected_model"] = find_model(context["selected_model_key"], context["model_catalog"])
        context["success_message"] = "Pipeline cargado correctamente."
    except Exception as exc:
        context["error_message"] = safe_text(exc)
        context["architecture"] = architecture_summary(context["selected_model_key"])

    return templates.TemplateResponse("pipeline.html", context)


@app.get("/system", response_class=HTMLResponse)
def system_page(request: Request) -> HTMLResponse:
    context = base_context(
        request,
        "Sistema y modelos",
        "Supervision del estado operativo y de la informacion disponible para cada modelo recurrente.",
    )
    arch = architecture_summary(context["selected_model_key"])
    model_info = metadata_summary(context["selected_model_key"])
    context["architecture"] = arch
    context["metadata_table_html"] = metadata_table(model_info)
    context["model_info"] = model_info
    context["model_catalog_table_html"] = model_catalog_table(context["model_catalog"])
    return templates.TemplateResponse("system.html", context)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
