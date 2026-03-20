"""
Página Acerca del Proyecto — Documentación del TFG.

Incluye información sobre:
  - Dataset MIT-BIH Arrhythmia Database
  - Clasificación AAMI y mapeo de clases
  - Arquitectura del sistema (frontend ↔ backend ↔ modelo)
  - Métricas del modelo (si metadata disponible)
  - Referencias bibliográficas
"""

import streamlit as st
import requests
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import apply_styles, glass_card

# ─── Configuración ────────────────────────────────────────────────────

st.set_page_config(page_title="Acerca del Proyecto | AEW", page_icon="📊", layout="wide")
apply_styles()

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# ─── Header ───────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 1rem 0;">
    <h1 style="font-size: 2.2rem;">📊 Acerca del Proyecto</h1>
    <p style="color: #8b949e; font-size: 1.05rem;">
        Trabajo de Fin de Grado — Detección temprana de arritmias cardíacas
        mediante redes neuronales recurrentes.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Dataset ──────────────────────────────────────────────────────────

st.markdown("### 🗃️ Dataset: MIT-BIH Arrhythmia Database")

st.markdown(glass_card("""
    <p style="color: #e6edf3; line-height: 1.7;">
        La <strong>MIT-BIH Arrhythmia Database</strong> es uno de los estándares de referencia
        en investigación de arritmias cardíacas. Contiene <strong>48 registros</strong> de ECG
        ambulatorio de 30 minutos cada uno, con una frecuencia de muestreo de 360 Hz y
        resolución de 11 bits. Cada latido está anotado por cardiólogos expertos.
    </p>
    <p style="color: #8b949e; margin-top: 0.8rem;">
        📎 Fuente:
        <a href="https://physionet.org/content/mitdb/1.0.0/" target="_blank"
           style="color: #58a6ff;">PhysioNet — MIT-BIH Arrhythmia Database</a>
    </p>
"""), unsafe_allow_html=True)

st.markdown("---")

# ─── Clasificación AAMI ──────────────────────────────────────────────

st.markdown("### 🏷️ Clasificación AAMI")

st.markdown("""
<p style="color: #8b949e;">
    Los latidos se clasifican según el estándar <strong>AAMI EC57</strong> en 5 superclases,
    agrupando los tipos de anotación originales del MIT-BIH.
</p>
""", unsafe_allow_html=True)

aami_data = {
    "Clase AAMI": ["Normal (N)", "SVEB (S)", "VEB (V)", "Fusion (F)", "Unknown (Q)"],
    "Descripción": [
        "Latido normal sinusal",
        "Latido ectópico supraventricular prematuro",
        "Latido ectópico ventricular prematuro",
        "Fusión de latido normal y ventricular",
        "Latido no clasificable o marcapasos",
    ],
    "Anotaciones MIT-BIH": [
        "N, L, R, e, j",
        "A, a, J, S",
        "V, E",
        "F",
        "/, f, Q",
    ],
    "Riesgo Clínico": [
        "🟢 Bajo",
        "🟡 Medio",
        "🔴 Alto",
        "🟡 Medio",
        "⚪ Indeterminado",
    ],
}

import pandas as pd
df_aami = pd.DataFrame(aami_data)
st.dataframe(df_aami, use_container_width=True, hide_index=True)

st.markdown("---")

# ─── Arquitectura del Sistema ─────────────────────────────────────────

st.markdown("### 🏗️ Arquitectura del Sistema")

st.markdown(glass_card("""
    <div style="font-family: monospace; color: #e6edf3; line-height: 2;">
        <div style="text-align: center;">
            <span style="background: rgba(0,212,255,0.15); padding: 0.5rem 1.5rem;
                         border-radius: 10px; border: 1px solid rgba(0,212,255,0.3);
                         font-weight: 600; color: #00d4ff;">
                🖥️ Frontend (Streamlit)
            </span>
            <br/>
            <span style="color: #30363d; font-size: 1.5rem;">│</span><br/>
            <span style="color: #8b949e;">HTTP / REST API</span><br/>
            <span style="color: #30363d; font-size: 1.5rem;">│</span><br/>
            <span style="background: rgba(188,140,255,0.15); padding: 0.5rem 1.5rem;
                         border-radius: 10px; border: 1px solid rgba(188,140,255,0.3);
                         font-weight: 600; color: #bc8cff;">
                ⚙️ Backend (FastAPI)
            </span>
            <br/>
            <span style="color: #30363d; font-size: 1.5rem;">│</span><br/>
            <span style="color: #8b949e;">TensorFlow / Keras</span><br/>
            <span style="color: #30363d; font-size: 1.5rem;">│</span><br/>
            <span style="background: rgba(63,185,80,0.15); padding: 0.5rem 1.5rem;
                         border-radius: 10px; border: 1px solid rgba(63,185,80,0.3);
                         font-weight: 600; color: #3fb950;">
                🧠 Modelo CNN+BiLSTM
            </span>
        </div>
    </div>
""", accent=True), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(glass_card("""
        <h4 style="color: #00d4ff; margin-top: 0;">Frontend</h4>
        <ul style="color: #8b949e; line-height: 1.8;">
            <li>Streamlit (Python)</li>
            <li>Plotly (visualizaciones)</li>
            <li>4 páginas multi-app</li>
            <li>Diseño responsivo</li>
        </ul>
    """), unsafe_allow_html=True)

with col2:
    st.markdown(glass_card("""
        <h4 style="color: #bc8cff; margin-top: 0;">Backend</h4>
        <ul style="color: #8b949e; line-height: 1.8;">
            <li>FastAPI (REST API)</li>
            <li>Pydantic (validación)</li>
            <li>8 endpoints</li>
            <li>Dockerizado</li>
        </ul>
    """), unsafe_allow_html=True)

with col3:
    st.markdown(glass_card("""
        <h4 style="color: #3fb950; margin-top: 0;">Modelo ML</h4>
        <ul style="color: #8b949e; line-height: 1.8;">
            <li>TensorFlow / Keras</li>
            <li>Conv1D + BiLSTM</li>
            <li>XAI por oclusión</li>
            <li>ECG sintético</li>
        </ul>
    """), unsafe_allow_html=True)

st.markdown("---")

# ─── Métricas del modelo ─────────────────────────────────────────────

st.markdown("### 📈 Métricas del Modelo")

try:
    info_res = requests.get(f"{BACKEND_URL}/model-info", timeout=3)
    if info_res.status_code == 200:
        info = info_res.json()
        meta = info.get("metadata", {})

        if "accuracy" in meta:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Accuracy", f"{meta['accuracy']:.2%}")
            with m2:
                st.metric("Macro F1-Score", f"{meta.get('macro_f1', 0):.2%}")
            with m3:
                st.metric("Balanced Accuracy", f"{meta.get('balanced_accuracy', 0):.2%}")

            if "confusion_matrix" in meta:
                import numpy as np
                import plotly.figure_factory as ff

                cm = np.array(meta["confusion_matrix"])
                classes = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"]

                fig_cm = ff.create_annotated_heatmap(
                    cm,
                    x=classes,
                    y=classes,
                    colorscale=[[0, "#0d1117"], [0.5, "#0969da"], [1, "#00d4ff"]],
                    showscale=True,
                )

                fig_cm.update_layout(
                    title=dict(text="Matriz de Confusión", font=dict(color="#e6edf3", size=14)),
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    font=dict(family="Inter", color="#e6edf3"),
                    height=450,
                    xaxis=dict(title="Predicción", color="#8b949e"),
                    yaxis=dict(title="Real", color="#8b949e", autorange="reversed"),
                )

                st.plotly_chart(fig_cm, use_container_width=True)

            if "class_distribution_train" in meta:
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    train_dist = meta["class_distribution_train"]
                    st.markdown("**Distribución de clases (Train):**")
                    for k, v in sorted(train_dist.items()):
                        class_name = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"][int(k)]
                        st.write(f"- {class_name}: {v:,}")
                with dist_col2:
                    if "class_distribution_test" in meta:
                        test_dist = meta["class_distribution_test"]
                        st.markdown("**Distribución de clases (Test):**")
                        for k, v in sorted(test_dist.items()):
                            class_name = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"][int(k)]
                            st.write(f"- {class_name}: {v:,}")
        else:
            st.info("ℹ️ El modelo está en **modo demo**. Las métricas estarán disponibles tras entrenar el modelo con datos reales.")
            st.markdown("""
            **Métricas esperadas tras entrenamiento:**
            - Accuracy: ~95%
            - Macro F1: ~75% (clases desbalanceadas)
            - Balanced Accuracy: ~80%
            """)
    else:
        st.warning("No se pudo obtener la información del modelo.")
except requests.exceptions.ConnectionError:
    st.warning("Backend no conectado. Inicia el backend para ver las métricas.")
except Exception:
    st.warning("No se pudieron cargar las métricas.")

st.markdown("---")

# ─── Endpoints API ────────────────────────────────────────────────────

st.markdown("### 🔌 Endpoints de la API")

endpoints_data = {
    "Método": ["GET", "GET", "GET", "POST", "POST", "POST", "POST", "POST"],
    "Ruta": [
        "/health", "/model-info", "/model-architecture",
        "/predict", "/explain", "/predict-and-explain",
        "/ecg-signal", "/pipeline-steps"
    ],
    "Descripción": [
        "Estado del sistema y modelo",
        "Información general del modelo y metadata",
        "Arquitectura detallada (capas, parámetros)",
        "Predicción de clase de arritmia",
        "Explicabilidad por oclusión de timesteps",
        "Predicción + explicabilidad combinadas",
        "Generación de ECG sintético (imagen base64)",
        "Pasos intermedios del pipeline para visualización",
    ],
}

df_endpoints = pd.DataFrame(endpoints_data)
st.dataframe(df_endpoints, use_container_width=True, hide_index=True)

st.markdown("---")

# ─── Referencias ──────────────────────────────────────────────────────

st.markdown("### 📚 Referencias Bibliográficas")

st.markdown("""
1. **Moody, G. B., & Mark, R. G.** (2001). The impact of the MIT-BIH Arrhythmia Database.
   *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.

2. **ANSI/AAMI EC57:2012**. Testing and reporting performance results of cardiac rhythm
   and ST segment measurement algorithms.

3. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory.
   *Neural Computation*, 9(8), 1735-1780.

4. **Schuster, M., & Paliwal, K. K.** (1997). Bidirectional Recurrent Neural Networks.
   *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.

5. **de Chazal, P., O'Dwyer, M., & Reilly, R. B.** (2004). Automatic classification of
   heartbeats using ECG morphology and heartbeat interval features.
   *IEEE Transactions on Biomedical Engineering*, 51(7), 1196-1206.

6. **Goldberger, A. L., et al.** (2000). PhysioBank, PhysioToolkit, and PhysioNet:
   Components of a New Research Resource for Complex Physiologic Signals.
   *Circulation*, 101(23), e215-e220.
""")

# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ℹ️ Información")
    st.markdown("""
    **Trabajo de Fin de Grado**

    Detección temprana de arritmias
    cardíacas con redes neuronales
    recurrentes.

    **Tecnologías:**
    - Python 3.11
    - TensorFlow / Keras
    - FastAPI
    - Streamlit
    - Docker
    """)
