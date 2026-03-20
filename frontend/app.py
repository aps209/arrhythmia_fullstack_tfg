"""
app.py — Dashboard principal de la plataforma Arrhythmia Early Warning.

Esta es la página de inicio de la aplicación multi-página Streamlit.
Muestra el estado del sistema, métricas clave y navegación visual.
"""

import streamlit as st
import requests
import os

from styles import apply_styles, glass_card

# ─── Configuración de página ──────────────────────────────────────────

st.set_page_config(
    page_title="Arrhythmia Early Warning",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# ─── Header ───────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <h1 style="font-size: 2.8rem; margin-bottom: 0.3rem;">
        ❤️ Arrhythmia Early Warning Platform
    </h1>
    <p style="color: #8b949e; font-size: 1.15rem; max-width: 700px; margin: 0 auto;">
        Plataforma de detección temprana de arritmias cardíacas basada en
        redes neuronales recurrentes (BiLSTM) a partir de intervalos R-R.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Estado del sistema ───────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

backend_connected = False
health_data = {}

try:
    health_res = requests.get(f"{BACKEND_URL}/health", timeout=3)
    if health_res.status_code == 200:
        backend_connected = True
        health_data = health_res.json()
except Exception:
    pass

with col1:
    status = "🟢 Conectado" if backend_connected else "🔴 Desconectado"
    st.metric("Estado Backend", status)

with col2:
    mode = health_data.get("model_mode", "—")
    mode_label = "🧠 Entrenado" if mode == "trained" else "🎲 Demo"
    st.metric("Modo del Modelo", mode_label)

with col3:
    version = health_data.get("model_version", "—")
    st.metric("Versión", version)

with col4:
    st.metric("Input Esperado", "15 R-R (seg)")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Secciones de navegación ──────────────────────────────────────────

st.subheader("🗂️ Secciones de la Plataforma")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown(glass_card("""
        <h3 style="color: #00d4ff; margin-top: 0;">🔬 Predicción</h3>
        <p style="color: #8b949e;">
            Introduce una secuencia de 15 intervalos R-R y obtén una predicción
            de arritmia con visualización ECG, nivel de riesgo y explicabilidad.
        </p>
        <p style="color: #58a6ff; font-size: 0.85rem;">
            → Navega desde el menú lateral
        </p>
    """, accent=True), unsafe_allow_html=True)

with nav_col2:
    st.markdown(glass_card("""
        <h3 style="color: #bc8cff; margin-top: 0;">🧠 Red Neuronal</h3>
        <p style="color: #8b949e;">
            Visualiza paso a paso cómo la red neuronal procesa los intervalos R-R:
            derivación de features, normalización y arquitectura del modelo.
        </p>
        <p style="color: #58a6ff; font-size: 0.85rem;">
            → Navega desde el menú lateral
        </p>
    """, accent=True), unsafe_allow_html=True)

with nav_col3:
    st.markdown(glass_card("""
        <h3 style="color: #3fb950; margin-top: 0;">📊 Acerca del Proyecto</h3>
        <p style="color: #8b949e;">
            Documentación del dataset, arquitectura del sistema, métricas
            del modelo y referencias bibliográficas del TFG.
        </p>
        <p style="color: #58a6ff; font-size: 0.85rem;">
            → Navega desde el menú lateral
        </p>
    """, accent=True), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Info del modelo ──────────────────────────────────────────────────

with st.expander("📋 Detalles técnicos del modelo", expanded=False):
    if backend_connected:
        try:
            arch_res = requests.get(f"{BACKEND_URL}/model-architecture", timeout=5)
            if arch_res.status_code == 200:
                arch = arch_res.json()

                t1, t2, t3 = st.columns(3)
                with t1:
                    st.metric("Parámetros Totales", f"{arch['total_params']:,}")
                with t2:
                    st.metric("Parámetros Entrenables", f"{arch['trainable_params']:,}")
                with t3:
                    st.metric("Nº Capas", len(arch["layers"]))

                st.markdown("**Arquitectura:**")
                for layer in arch["layers"]:
                    trainable = "✅" if layer["trainable"] else "—"
                    st.markdown(
                        f"- `{layer['name']}` ({layer['type']}) → "
                        f"{layer['output_shape']} | "
                        f"{layer['num_params']:,} params | Entrenable: {trainable}"
                    )
            else:
                st.warning("No se pudo obtener la arquitectura del modelo.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Conecta al backend para ver los detalles del modelo.")

# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #e6edf3; font-size: 1.3rem;">❤️ AEW Platform</h2>
        <p style="color: #8b949e; font-size: 0.8rem;">Arrhythmia Early Warning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔧 Configuración")
    st.text_input("Backend URL", value=BACKEND_URL, disabled=True)

    if backend_connected:
        st.success("🟢 Backend conectado")
    else:
        st.error("🔴 Backend desconectado")

    st.markdown("---")
    st.markdown("""
    <p style="color: #8b949e; font-size: 0.75rem; text-align: center;">
        TFG — Detección de Arritmias<br/>
        CNN + BiLSTM | MIT-BIH Database
    </p>
    """, unsafe_allow_html=True)
