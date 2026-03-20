"""
styles.py — Sistema de diseño centralizado para el frontend Streamlit.

Proporciona CSS premium con glassmorphism, gradientes, animaciones
y una paleta de colores coherente para todas las páginas.
"""


def get_custom_css() -> str:
    """Devuelve el CSS personalizado inyectable en cualquier página Streamlit."""
    return """
    <style>
    /* ─── Imports ────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Variables CSS ──────────────────────────────────── */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-card: rgba(22, 27, 34, 0.85);
        --border-color: rgba(48, 54, 61, 0.6);
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --accent-cyan: #00d4ff;
        --accent-blue: #58a6ff;
        --accent-green: #3fb950;
        --accent-red: #ff6b6b;
        --accent-orange: #f0883e;
        --accent-purple: #bc8cff;
        --gradient-main: linear-gradient(135deg, #00d4ff 0%, #0969da 50%, #bc8cff 100%);
        --gradient-card: linear-gradient(145deg, rgba(22,27,34,0.9), rgba(13,17,23,0.95));
        --glass-bg: rgba(22, 27, 34, 0.6);
        --glass-border: rgba(255, 255, 255, 0.08);
        --shadow-glow: 0 0 30px rgba(0, 212, 255, 0.1);
    }

    /* ─── Global ─────────────────────────────────────────── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    /* ─── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary) !important;
    }

    /* ─── Headers ────────────────────────────────────────── */
    h1 {
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* ─── Metric Cards ───────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: var(--shadow-glow);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 40px rgba(0, 212, 255, 0.15);
        border-color: rgba(0, 212, 255, 0.3);
    }

    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--accent-cyan) !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }

    /* ─── Glass Cards ────────────────────────────────────── */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.8rem;
        backdrop-filter: blur(12px);
        box-shadow: var(--shadow-glow);
        margin-bottom: 1.5rem;
    }

    .glass-card-accent {
        background: linear-gradient(145deg, rgba(0, 212, 255, 0.05), rgba(22, 27, 34, 0.85));
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 1.8rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.08);
        margin-bottom: 1.5rem;
    }

    /* ─── Buttons ────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #0969da, #00d4ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.25) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4) !important;
    }

    /* ─── Form ───────────────────────────────────────────── */
    .stForm {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(12px) !important;
    }

    /* ─── Text Inputs ────────────────────────────────────── */
    .stTextArea textarea, .stTextInput input {
        background: rgba(13, 17, 23, 0.8) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', monospace !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15) !important;
    }

    /* ─── Alerts ─────────────────────────────────────────── */
    .stAlert {
        border-radius: 12px !important;
        backdrop-filter: blur(8px) !important;
    }

    /* ─── Tabs ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0969da, #00d4ff) !important;
        color: white !important;
    }

    /* ─── Expanders ──────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        border-radius: 12px !important;
        border: 1px solid var(--glass-border) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    /* ─── Dataframes / Tables ────────────────────────────── */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ─── Animations ─────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.1); }
        50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.25); }
    }

    .animate-fade-in {
        animation: fadeInUp 0.6s ease forwards;
    }

    .pulse-glow {
        animation: pulse-glow 3s ease-in-out infinite;
    }

    /* ─── Risk Badges ────────────────────────────────────── */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .risk-low {
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }

    .risk-medium {
        background: rgba(240, 136, 62, 0.15);
        color: #f0883e;
        border: 1px solid rgba(240, 136, 62, 0.3);
    }

    .risk-high {
        background: rgba(255, 107, 107, 0.15);
        color: #ff6b6b;
        border: 1px solid rgba(255, 107, 107, 0.3);
    }

    /* ─── Pipeline Step Cards ────────────────────────────── */
    .pipeline-step {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        position: relative;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }

    .pipeline-step:hover {
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateX(4px);
    }

    .pipeline-step::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: var(--gradient-main);
        border-radius: 3px 0 0 3px;
    }

    .step-number {
        display: inline-block;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        border-radius: 50%;
        background: linear-gradient(135deg, #0969da, #00d4ff);
        color: white;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.6rem;
    }

    /* ─── Scrollbar ──────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    </style>
    """


def apply_styles():
    """Aplica los estilos CSS personalizados usando st.markdown."""
    import streamlit as st
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def risk_badge(risk_level: str) -> str:
    """Genera HTML para un badge de nivel de riesgo."""
    css_class = f"risk-{risk_level.lower()}"
    labels = {"low": "🟢 BAJO", "medium": "🟡 MEDIO", "high": "🔴 ALTO"}
    label = labels.get(risk_level.lower(), risk_level.upper())
    return f'<div class="risk-badge {css_class}">{label}</div>'


def glass_card(content: str, accent: bool = False) -> str:
    """Envuelve contenido HTML en una tarjeta glass."""
    cls = "glass-card-accent" if accent else "glass-card"
    return f'<div class="{cls}">{content}</div>'


def pipeline_step_card(step_num: int, title: str, description: str) -> str:
    """Genera HTML para un paso del pipeline."""
    return f"""
    <div class="pipeline-step">
        <span class="step-number">{step_num}</span>
        <strong style="color: var(--text-primary);">{title}</strong>
        <p style="color: var(--text-secondary); margin: 0.4rem 0 0 2.2rem; font-size: 0.9rem;">
            {description}
        </p>
    </div>
    """
