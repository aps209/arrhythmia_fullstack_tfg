# ❤️ Arrhythmia Early Warning Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Trabajo de Fin de Grado** — Plataforma full-stack para detección temprana de arritmias cardíacas a partir de ventanas de 15 intervalos R-R, utilizando redes neuronales recurrentes bidireccionales (CNN + BiLSTM).

---

## 📸 Características Principales

- 🔬 **Predicción de arritmias** en 5 clases AAMI (Normal, SVEB, VEB, Fusion, Unknown)
- 💓 **Visualización ECG sintética** generada a partir de intervalos R-R
- 🧠 **Pipeline visual** paso a paso de la red neuronal
- 📊 **Explicabilidad (XAI)** por oclusión de timesteps
- 🎨 **Interfaz premium** con diseño glassmorphism y modo oscuro
- 🐳 **Dockerizado** para despliegue sencillo

---

## 🏗️ Arquitectura del Sistema

```
┌──────────────────────┐     HTTP/REST     ┌──────────────────────┐
│                      │ ◄──────────────── │                      │
│   Frontend           │                   │   Backend            │
│   (Streamlit)        │ ──────────────► │   (FastAPI)          │
│                      │                   │                      │
│   • Dashboard        │                   │   • /predict         │
│   • Predicción + ECG │                   │   • /explain         │
│   • Red Neuronal     │                   │   • /ecg-signal      │
│   • Documentación    │                   │   • /pipeline-steps  │
│                      │                   │   • /model-arch      │
└──────────────────────┘                   └──────────┬───────────┘
                                                      │
                                           ┌──────────▼───────────┐
                                           │   Modelo ML          │
                                           │   (TensorFlow/Keras) │
                                           │                      │
                                           │   Conv1D + BiLSTM    │
                                           │   → Softmax (5)      │
                                           └──────────────────────┘
```

---

## 🔌 API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del sistema |
| `GET` | `/model-info` | Metadata del modelo |
| `GET` | `/model-architecture` | Arquitectura detallada (capas, parámetros) |
| `POST` | `/predict` | Predicción de clase de arritmia |
| `POST` | `/explain` | Explicabilidad por oclusión |
| `POST` | `/predict-and-explain` | Predicción + explicabilidad combinadas |
| `POST` | `/ecg-signal` | Generación de ECG sintético (PNG base64) |
| `POST` | `/pipeline-steps` | Pasos intermedios del pipeline |

---

## 📂 Estructura del Proyecto

```
arrhythmia_fullstack_tfg/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuración y rutas
│   │   ├── ecg_generator.py   # Generador de ECG sintético
│   │   ├── main.py            # Aplicación FastAPI
│   │   ├── model_service.py   # Servicio de inferencia y XAI
│   │   └── schemas.py         # Modelos Pydantic
│   ├── tests/
│   │   └── test_api.py        # Tests de integración
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── pages/
│   │   ├── 1_🔬_Prediccion.py       # Predicción + ECG
│   │   ├── 2_🧠_Red_Neuronal.py     # Pipeline visual
│   │   └── 3_📊_Acerca_del_Proyecto.py  # Documentación
│   ├── app.py                 # Dashboard principal
│   ├── styles.py              # Sistema de diseño CSS
│   ├── Dockerfile
│   └── requirements.txt
├── train/
│   ├── dataset_builder.py     # Constructor del dataset MIT-BIH
│   └── train_patient_split.py # Entrenamiento con GroupShuffleSplit
├── artifacts/                 # model.keras, scaler.joblib, metadata.json
├── docker-compose.yml
└── README.md
```

---

## 🚀 Cómo levantar el proyecto

### Opción 1: Docker Compose (recomendado)

```bash
docker-compose up --build
```

- Frontend: **http://localhost:8501**
- Backend: **http://localhost:8000**
- API Docs: **http://localhost:8000/docs**

### Opción 2: Ejecución local

**Terminal 1 — Backend:**
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -r requirements.txt
set BACKEND_URL=http://127.0.0.1:8000
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

---

## 🧪 Tests

```bash
cd backend
python -m pytest tests/test_api.py -v
```

---

## 🧠 Pipeline de Machine Learning

1. **Datos**: MIT-BIH Arrhythmia Database (PhysioNet)
2. **Segmentación**: Ventanas de 15 intervalos R-R consecutivos
3. **Features**: 4 canales (R-R, ΔR-R, Media Móvil, Z-Score)
4. **Normalización**: StandardScaler por feature
5. **Modelo**: Conv1D(32) → BN → BiLSTM(64) → BiLSTM(32) → Dense(64) → Softmax(5)
6. **Validación**: GroupShuffleSplit por paciente (sin data leakage)
7. **Clases**: 5 superclases AAMI (N, SVEB, VEB, F, Q)

---

## 📚 Referencias

- Moody, G. B., & Mark, R. G. (2001). The Impact of the MIT-BIH Arrhythmia Database.
- ANSI/AAMI EC57:2012. Testing and Reporting Performance Results.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
- de Chazal, P., et al. (2004). Automatic Classification of Heartbeats.

---

## 📄 Licencia

Proyecto académico — Trabajo de Fin de Grado.
