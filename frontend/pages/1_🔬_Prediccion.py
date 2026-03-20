"""
Página de Predicción — Análisis de arritmias con visualización ECG.

Permite al usuario introducir 15 intervalos R-R y obtener:
  - Señal ECG sintética con picos R marcados
  - Predicción de clase de arritmia con nivel de riesgo
  - Explicabilidad: timesteps más importantes
  - Gráficos interactivos con Plotly
"""

import streamlit as st
import requests
import os
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles, risk_badge, glass_card

# ─── Configuración ────────────────────────────────────────────────────

st.set_page_config(page_title="Predicción | AEW", page_icon="🔬", layout="wide")
apply_styles()

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# ─── Header ───────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 1rem 0;">
    <h1 style="font-size: 2.2rem;">🔬 Análisis y Predicción</h1>
    <p style="color: #8b949e; font-size: 1.05rem;">
        Introduce 15 intervalos R-R para obtener la predicción de arritmia,
        visualización ECG sintética y análisis de explicabilidad.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Secuencias de ejemplo ────────────────────────────────────────────

EXAMPLES = {
    "Ritmo Normal": "0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.85",
    "Posible Arritmia (VEB)": "0.82, 0.79, 0.81, 0.45, 1.20, 0.78, 0.80, 0.42, 1.18, 0.81, 0.79, 0.43, 1.15, 0.80, 0.78",
    "Taquicardia": "0.45, 0.44, 0.46, 0.43, 0.45, 0.44, 0.43, 0.46, 0.44, 0.45, 0.43, 0.44, 0.46, 0.45, 0.44",
    "Bradicardia": "1.20, 1.22, 1.18, 1.21, 1.19, 1.23, 1.20, 1.18, 1.22, 1.21, 1.19, 1.20, 1.23, 1.18, 1.21",
    "Arritmia con latido prematuro": "0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.84, 0.85, 0.86, 0.60",
}

st.markdown("### 📋 Secuencias de ejemplo")
example_cols = st.columns(len(EXAMPLES))
selected_example = None

for i, (name, seq) in enumerate(EXAMPLES.items()):
    with example_cols[i]:
        if st.button(name, key=f"example_{i}", use_container_width=True):
            selected_example = seq

# ─── Formulario de entrada ────────────────────────────────────────────

st.markdown("### 📝 Entrada de datos")

default_rrs = selected_example or list(EXAMPLES.values())[0]

with st.form("predict_form"):
    rr_input = st.text_area(
        "Intervalos R-R (15 valores en segundos, separados por comas)",
        value=default_rrs,
        height=80,
        help="Cada valor representa la duración en segundos entre dos picos R consecutivos del ECG."
    )
    submit_button = st.form_submit_button("🔬 Analizar Secuencia", use_container_width=True)

# ─── Procesamiento y resultados ───────────────────────────────────────

if submit_button:
    try:
        rr_list = [float(x.strip()) for x in rr_input.split(",") if x.strip()]

        if len(rr_list) != 15:
            st.error(f"❌ Se requieren exactamente 15 intervalos R-R. Se proporcionaron {len(rr_list)}.")
        elif any(v <= 0 or v > 5 for v in rr_list):
            st.error("❌ Cada intervalo R-R debe estar entre 0 y 5 segundos.")
        else:
            with st.spinner("⏳ Analizando secuencia..."):
                payload = {"rr_intervals": rr_list}

                # Llamadas paralelas al backend
                pred_response = requests.post(f"{BACKEND_URL}/predict-and-explain", json=payload, timeout=15)
                ecg_response = requests.post(f"{BACKEND_URL}/ecg-signal", json=payload, timeout=15)

            if pred_response.status_code == 200:
                data = pred_response.json()
                pred = data["prediction"]
                exp = data["explanation"]

                st.markdown("---")

                # ═══ Señal ECG ═══════════════════════════════════════
                st.markdown("### 💓 Electrocardiograma Sintético")

                if ecg_response.status_code == 200:
                    ecg_data = ecg_response.json()
                    ecg_b64 = ecg_data["ecg_image_base64"]

                    # Mostrar imagen ECG
                    st.image(
                        base64.b64decode(ecg_b64),
                        use_column_width="always",
                    )

                    # Métricas del ECG
                    ecg_c1, ecg_c2, ecg_c3, ecg_c4 = st.columns(4)
                    with ecg_c1:
                        st.metric("Duración", f"{ecg_data['duration_seconds']:.1f} s")
                    with ecg_c2:
                        st.metric("Nº Latidos", ecg_data["num_beats"])
                    with ecg_c3:
                        st.metric("FC Media", f"{ecg_data['mean_heart_rate_bpm']:.0f} bpm")
                    with ecg_c4:
                        st.metric("Muestreo", f"{ecg_data['fs']} Hz")
                else:
                    st.warning("⚠️ No se pudo generar la señal ECG.")

                st.markdown("---")

                # ═══ Resultados de predicción ════════════════════════
                st.markdown("### 🎯 Resultado del Diagnóstico")

                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    # Predicción principal
                    st.markdown(glass_card(f"""
                        <div style="text-align: center;">
                            <p style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;
                                      letter-spacing: 0.1em; margin-bottom: 0.5rem;">Clase Predicha</p>
                            <h2 style="color: #e6edf3; font-size: 2rem; margin: 0.3rem 0;">
                                {pred['class_name']}
                            </h2>
                            <p style="color: #00d4ff; font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">
                                {pred['confidence'] * 100:.1f}% confianza
                            </p>
                            {risk_badge(pred['risk_level'])}
                        </div>
                    """, accent=True), unsafe_allow_html=True)

                    # Texto clínico
                    st.markdown(glass_card(f"""
                        <p style="color: #8b949e; font-size: 0.85rem; text-transform: uppercase;
                                  letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                            📋 Interpretación Clínica
                        </p>
                        <p style="color: #e6edf3; line-height: 1.6;">
                            {exp['clinical_text']}
                        </p>
                    """), unsafe_allow_html=True)

                with res_col2:
                    # Gráfico de probabilidades
                    probs = pred["probabilities"]
                    probs_df = pd.DataFrame([
                        {"Clase": k, "Probabilidad": v}
                        for k, v in probs.items()
                    ]).sort_values("Probabilidad", ascending=True)

                    colors = ["#3fb950" if c == pred["class_name"] else "#30363d"
                              for c in probs_df["Clase"]]

                    fig_probs = go.Figure(go.Bar(
                        y=probs_df["Clase"],
                        x=probs_df["Probabilidad"],
                        orientation="h",
                        marker=dict(
                            color=colors,
                            line=dict(color="rgba(255,255,255,0.1)", width=1),
                        ),
                        text=[f"{v:.1%}" for v in probs_df["Probabilidad"]],
                        textposition="outside",
                        textfont=dict(color="#e6edf3", size=12),
                    ))

                    fig_probs.update_layout(
                        title=dict(text="Distribución de Probabilidades", font=dict(color="#e6edf3", size=14)),
                        xaxis=dict(title="Probabilidad", range=[0, 1.05],
                                   gridcolor="#1a2332", color="#8b949e"),
                        yaxis=dict(color="#8b949e"),
                        plot_bgcolor="#0d1117",
                        paper_bgcolor="#0d1117",
                        font=dict(family="Inter", color="#e6edf3"),
                        height=350,
                        margin=dict(l=10, r=30, t=40, b=10),
                    )
                    st.plotly_chart(fig_probs, use_container_width=True)

                st.markdown("---")

                # ═══ Secuencia R-R con explicabilidad ════════════════
                st.markdown("### 📈 Secuencia R-R con Explicabilidad")

                fig_rr = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.6, 0.4],
                    shared_xaxes=True,
                    subplot_titles=("Intervalos R-R", "Importancia por Timestep (XAI)"),
                    vertical_spacing=0.12,
                )

                # Gráfico superior: R-R intervals
                fig_rr.add_trace(go.Scatter(
                    x=list(range(1, len(rr_list) + 1)),
                    y=rr_list,
                    mode="lines+markers",
                    name="R-R Interval",
                    line=dict(color="#00d4ff", width=2.5),
                    marker=dict(size=8, color="#00d4ff",
                                line=dict(color="#0d1117", width=1.5)),
                ), row=1, col=1)

                # Destacar timesteps importantes
                important_idx = [t - 1 for t in exp["top_timesteps"] if t <= len(rr_list)]
                important_rrs = [rr_list[i] for i in important_idx]
                important_x = [i + 1 for i in important_idx]

                fig_rr.add_trace(go.Scatter(
                    x=important_x,
                    y=important_rrs,
                    mode="markers",
                    name="Alta Importancia",
                    marker=dict(size=16, color="#ff6b6b", symbol="star",
                                line=dict(color="white", width=1)),
                ), row=1, col=1)

                # Gráfico inferior: Importancia
                importance_colors = [
                    "#ff6b6b" if v > 0.15 else "#f0883e" if v > 0.08 else "#30363d"
                    for v in exp["timestep_importance"]
                ]

                fig_rr.add_trace(go.Bar(
                    x=list(range(1, len(exp["timestep_importance"]) + 1)),
                    y=exp["timestep_importance"],
                    name="Importancia",
                    marker=dict(color=importance_colors,
                                line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
                    text=[f"{v:.1%}" for v in exp["timestep_importance"]],
                    textposition="outside",
                    textfont=dict(color="#8b949e", size=9),
                ), row=2, col=1)

                fig_rr.update_layout(
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    font=dict(family="Inter", color="#e6edf3"),
                    height=550,
                    showlegend=True,
                    legend=dict(
                        bgcolor="rgba(22,27,34,0.8)",
                        bordercolor="#30363d",
                        borderwidth=1,
                        font=dict(color="#e6edf3"),
                    ),
                    margin=dict(l=10, r=10, t=30, b=10),
                )

                fig_rr.update_xaxes(gridcolor="#1a2332", color="#8b949e", title="Paso temporal")
                fig_rr.update_yaxes(gridcolor="#1a2332", color="#8b949e")
                fig_rr.update_yaxes(title="R-R (s)", row=1, col=1)
                fig_rr.update_yaxes(title="Importancia", row=2, col=1)

                st.plotly_chart(fig_rr, use_container_width=True)

                # ═══ Resumen estadístico ═════════════════════════════
                with st.expander("📊 Resumen estadístico del input", expanded=False):
                    summary = pred.get("input_summary", {})
                    s_cols = st.columns(5)
                    labels = [
                        ("Media R-R", "rr_mean", "s"),
                        ("Desv. Estándar", "rr_std", "s"),
                        ("Mínimo", "rr_min", "s"),
                        ("Máximo", "rr_max", "s"),
                        ("Rango", "rr_range", "s"),
                    ]
                    for i, (label, key, unit) in enumerate(labels):
                        with s_cols[i]:
                            val = summary.get(key, 0)
                            st.metric(label, f"{val:.4f} {unit}")

            else:
                st.error(f"❌ Error del backend: {pred_response.text}")

    except ValueError:
        st.error("❌ Entrada inválida. Asegúrate de introducir solo números separados por comas.")
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar al backend. ¿Está ejecutándose?")
    except requests.exceptions.Timeout:
        st.error("❌ El backend tardó demasiado en responder.")
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")

# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💡 Guía rápida")
    st.markdown("""
    1. Elige una secuencia de ejemplo o introduce la tuya
    2. Pulsa **Analizar Secuencia**
    3. Revisa el ECG, la predicción y la explicabilidad

    **¿Qué son los intervalos R-R?**

    Son las duraciones en segundos entre picos R
    consecutivos del electrocardiograma. Un adulto
    sano en reposo tiene R-R ≈ 0.6 – 1.0 s.
    """)
