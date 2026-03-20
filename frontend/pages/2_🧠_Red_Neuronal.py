"""
Página Red Neuronal — Visualización del pipeline de la red neuronal.

Muestra paso a paso cómo la red neuronal procesa los intervalos R-R:
  1. Datos crudos → Derivación de features
  2. Normalización con StandardScaler
  3. Arquitectura del modelo (Conv1D + BiLSTM + Dense)
  4. Salida softmax → Predicción
"""

import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles, glass_card, pipeline_step_card

# ─── Configuración ────────────────────────────────────────────────────

st.set_page_config(page_title="Red Neuronal | AEW", page_icon="🧠", layout="wide")
apply_styles()

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# ─── Header ───────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 1rem 0;">
    <h1 style="font-size: 2.2rem;">🧠 Pipeline de la Red Neuronal</h1>
    <p style="color: #8b949e; font-size: 1.05rem;">
        Visualiza paso a paso cómo el modelo transforma los intervalos R-R
        en una predicción de arritmia. Ideal para entender el flujo de datos.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Diagrama del pipeline ────────────────────────────────────────────

st.markdown("### 🔄 Flujo de Procesamiento")

st.markdown(
    pipeline_step_card(1, "Entrada: 15 Intervalos R-R",
        "Secuencia de duraciones entre picos R consecutivos del ECG (en segundos)"),
    unsafe_allow_html=True
)
st.markdown(
    pipeline_step_card(2, "Derivación de Features (4 canales)",
        "Se calculan: ΔR-R (diferencia), Media Móvil (ventana=3) y Z-Score normalizado"),
    unsafe_allow_html=True
)
st.markdown(
    pipeline_step_card(3, "Normalización (StandardScaler)",
        "Los 60 valores (15×4) se normalizan con media=0, std=1 usando el scaler entrenado"),
    unsafe_allow_html=True
)
st.markdown(
    pipeline_step_card(4, "Conv1D (32 filtros, kernel=3)",
        "Convolución 1D extrae patrones locales de la secuencia temporal"),
    unsafe_allow_html=True
)
st.markdown(
    pipeline_step_card(5, "BatchNormalization + BiLSTM (64+64 units)",
        "Red LSTM bidireccional captura dependencias temporales a corto y largo plazo"),
    unsafe_allow_html=True
)
st.markdown(
    pipeline_step_card(6, "BiLSTM (32+32 units) + Dropout",
        "Segunda capa BiLSTM reduce dimensionalidad. Dropout(0.3) previene sobreajuste"),
    unsafe_allow_html=True
)
st.markdown(
    pipeline_step_card(7, "Dense(64, relu) + Dense(5, softmax)",
        "Capas fully-connected producen 5 probabilidades: Normal, SVEB, VEB, Fusion, Unknown"),
    unsafe_allow_html=True
)

st.markdown("---")

# ─── Visualización interactiva del pipeline ───────────────────────────

st.markdown("### 🔍 Explorar el Pipeline con Datos Reales")

default_rrs = "0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.60"

with st.form("pipeline_form"):
    rr_input = st.text_area(
        "Intervalos R-R (15 valores separados por comas)",
        value=default_rrs,
        height=70,
    )
    submit = st.form_submit_button("🔍 Visualizar Pipeline", use_container_width=True)

if submit:
    try:
        rr_list = [float(x.strip()) for x in rr_input.split(",") if x.strip()]

        if len(rr_list) != 15:
            st.error(f"Se requieren exactamente 15 intervalos. Se proporcionaron {len(rr_list)}.")
        else:
            with st.spinner("Procesando pipeline..."):
                payload = {"rr_intervals": rr_list}
                response = requests.post(f"{BACKEND_URL}/pipeline-steps", json=payload, timeout=10)

            if response.status_code == 200:
                pipeline = response.json()

                # ─── Tab 1: Features derivadas ────────────────────
                tab1, tab2, tab3 = st.tabs([
                    "📊 Features Derivadas",
                    "🏗️ Arquitectura del Modelo",
                    "🎯 Resultado"
                ])

                with tab1:
                    st.markdown("#### Transformación de Features")
                    st.markdown("""
                    <p style="color: #8b949e;">
                        A partir de los 15 R-R crudos, se derivan <strong>4 canales</strong>
                        de entrada para la red neuronal, creando un tensor de forma
                        <code>(1, 15, 4)</code>.
                    </p>
                    """, unsafe_allow_html=True)

                    features = pipeline["derived_features"]

                    # Crear gráfico con los 4 canales
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=[f["step_name"] for f in features],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.08,
                    )

                    colors = ["#00d4ff", "#ff6b6b", "#3fb950", "#bc8cff"]

                    for i, feat in enumerate(features):
                        row, col = (i // 2) + 1, (i % 2) + 1
                        fig.add_trace(go.Scatter(
                            x=list(range(1, 16)),
                            y=feat["values"],
                            mode="lines+markers",
                            name=feat["step_name"],
                            line=dict(color=colors[i], width=2),
                            marker=dict(size=6, color=colors[i]),
                            hovertemplate=f"Paso %{{x}}<br>{feat['step_name']}: %{{y:.4f}}<extra></extra>",
                        ), row=row, col=col)

                    fig.update_layout(
                        plot_bgcolor="#0d1117",
                        paper_bgcolor="#0d1117",
                        font=dict(family="Inter", color="#e6edf3"),
                        height=500,
                        showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )

                    fig.update_xaxes(gridcolor="#1a2332", color="#8b949e")
                    fig.update_yaxes(gridcolor="#1a2332", color="#8b949e")

                    st.plotly_chart(fig, use_container_width=True)

                    # Tabla de features
                    st.markdown("#### 📋 Tabla de Valores")
                    df_features = pd.DataFrame({
                        f["step_name"]: [round(v, 5) for v in f["values"]]
                        for f in features
                    })
                    df_features.index = [f"Paso {i+1}" for i in range(15)]
                    st.dataframe(df_features, use_container_width=True, height=400)

                with tab2:
                    st.markdown("#### Arquitectura del Modelo CNN + BiLSTM")

                    try:
                        arch_res = requests.get(f"{BACKEND_URL}/model-architecture", timeout=5)
                        if arch_res.status_code == 200:
                            arch = arch_res.json()

                            # Métricas
                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                st.metric("Parámetros Totales", f"{arch['total_params']:,}")
                            with m2:
                                st.metric("Entrenables", f"{arch['trainable_params']:,}")
                            with m3:
                                st.metric("Input Shape", arch["input_shape"])
                            with m4:
                                st.metric("Output Shape", arch["output_shape"])

                            st.markdown("<br>", unsafe_allow_html=True)

                            # Diagrama visual de la arquitectura
                            st.markdown("#### 📐 Diagrama de Capas")

                            layers = arch["layers"]

                            # Crear Sankey diagram del flujo
                            layer_names = [l["name"] for l in layers]
                            layer_types = [l["type"] for l in layers]
                            layer_params = [l["num_params"] for l in layers]

                            # Gráfico de barras de parámetros por capa
                            fig_arch = go.Figure()

                            # Filtrar capas con parámetros
                            filtered = [(l["name"], l["type"], l["num_params"], l.get("output_shape", ""))
                                       for l in layers if l["num_params"] > 0]

                            if filtered:
                                names, types, params, shapes = zip(*filtered)

                                fig_arch.add_trace(go.Bar(
                                    y=[f"{n}\n({t})" for n, t in zip(names, types)],
                                    x=list(params),
                                    orientation="h",
                                    marker=dict(
                                        color=params,
                                        colorscale=[[0, "#0969da"], [0.5, "#00d4ff"], [1, "#bc8cff"]],
                                        line=dict(color="rgba(255,255,255,0.1)", width=1),
                                    ),
                                    text=[f"{p:,} params" for p in params],
                                    textposition="outside",
                                    textfont=dict(color="#e6edf3", size=11),
                                    hovertemplate="<b>%{y}</b><br>Parámetros: %{x:,}<extra></extra>",
                                ))

                                fig_arch.update_layout(
                                    title=dict(text="Parámetros por Capa",
                                              font=dict(color="#e6edf3", size=14)),
                                    plot_bgcolor="#0d1117",
                                    paper_bgcolor="#0d1117",
                                    font=dict(family="Inter", color="#e6edf3"),
                                    height=400,
                                    margin=dict(l=10, r=80, t=40, b=10),
                                    xaxis=dict(gridcolor="#1a2332", color="#8b949e"),
                                    yaxis=dict(color="#8b949e", autorange="reversed"),
                                )

                                st.plotly_chart(fig_arch, use_container_width=True)

                            # Tabla detallada
                            st.markdown("#### 📋 Detalle de Capas")
                            df_layers = pd.DataFrame(layers)
                            df_layers.columns = ["Nombre", "Tipo", "Output Shape", "Parámetros", "Entrenable"]
                            df_layers["Entrenable"] = df_layers["Entrenable"].map({True: "✅", False: "—"})
                            st.dataframe(df_layers, use_container_width=True)

                            # Info del optimizer
                            if arch.get("optimizer"):
                                st.markdown(f"""
                                **Configuración de Entrenamiento:**
                                - Optimizador: `{arch['optimizer']}`
                                - Función de Pérdida: `{arch.get('loss_function', '—')}`
                                """)
                    except Exception as e:
                        st.error(f"Error obteniendo la arquitectura: {e}")

                with tab3:
                    st.markdown("#### Predicción Final")

                    probs = pipeline["prediction_probabilities"]
                    predicted = pipeline["predicted_class"]

                    st.markdown(glass_card(f"""
                        <div style="text-align: center;">
                            <p style="color: #8b949e; text-transform: uppercase;
                                      letter-spacing: 0.1em; font-size: 0.85rem;">Clase Predicha</p>
                            <h2 style="color: #00d4ff; font-size: 2.2rem; margin: 0.5rem 0;">
                                {predicted}
                            </h2>
                        </div>
                    """, accent=True), unsafe_allow_html=True)

                    # Gráfico gauge de probabilidades
                    fig_gauge = go.Figure()

                    class_colors = {
                        "Normal": "#3fb950",
                        "SVEB": "#f0883e",
                        "VEB": "#ff6b6b",
                        "Fusion": "#bc8cff",
                        "Unknown": "#8b949e",
                    }

                    for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        color = class_colors.get(cls, "#8b949e")
                        fig_gauge.add_trace(go.Bar(
                            x=[prob],
                            y=[cls],
                            orientation="h",
                            marker=dict(color=color, opacity=0.8),
                            text=f"{prob:.2%}",
                            textposition="outside",
                            textfont=dict(color="#e6edf3"),
                            name=cls,
                            showlegend=False,
                        ))

                    fig_gauge.update_layout(
                        plot_bgcolor="#0d1117",
                        paper_bgcolor="#0d1117",
                        font=dict(family="Inter", color="#e6edf3"),
                        height=300,
                        xaxis=dict(range=[0, 1.1], gridcolor="#1a2332",
                                   color="#8b949e", title="Probabilidad"),
                        yaxis=dict(color="#8b949e"),
                        margin=dict(l=10, r=50, t=10, b=10),
                        barmode="group",
                    )

                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.info(f"🔧 Modo del modelo: **{pipeline['model_mode']}**")

            else:
                st.error(f"Error del backend: {response.text}")

    except ValueError:
        st.error("Entrada inválida. Solo se aceptan números separados por comas.")
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar al backend.")
    except Exception as e:
        st.error(f"Error: {e}")

# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📖 Sobre el modelo")
    st.markdown("""
    **Arquitectura:** Conv1D + BiLSTM

    El modelo combina:
    - **Conv1D**: captura patrones locales
    - **BiLSTM**: captura dependencias
      temporales bidireccionales
    - **Dense + Softmax**: clasificación
      en 5 clases AAMI

    **Features de entrada (4 canales):**
    1. R-R interval original
    2. ΔR-R (diferencia)
    3. Media móvil (ventana 3)
    4. Z-score normalizado
    """)
