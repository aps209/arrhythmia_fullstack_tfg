# Arrhythmia ECG Workstation

Plataforma full-stack para análisis de arritmias sobre ECG real en formato WFDB. El sistema ya no trabaja con cadenas de intervalos R-R como entrada principal: consume registros `.dat + .hea` y utiliza `.atr` cuando existe para segmentación supervisada.

## Qué hace ahora

- Carga registros WFDB reales desde frontend y backend
- Segmenta latidos sobre ECG filtrado y remuestreado
- Clasifica cada segmento con un modelo `Conv1D + BiLSTM`
- Agrega probabilidades a nivel de registro
- Genera explicabilidad técnica basada en gradientes sobre la señal real
- Visualiza ECG bruto, ECG filtrado y regiones temporales relevantes

## Arquitectura

- Frontend: Gradio
- Backend: FastAPI
- Modelo: TensorFlow / Keras
- Backbone: `Conv1D + BiLSTM`
- Dataset objetivo: MIT-BIH Arrhythmia Database en formato WFDB

## Formato de entrada

Para inferencia se requiere al menos:

- `registro.dat`
- `registro.hea`

Opcional, pero recomendado:

- `registro.atr`

Nota: el fichero `.dat` por sí solo no basta. El `.hea` es obligatorio para interpretar correctamente el ECG.

## Entrenamiento

Instala dependencias:

```bash
cd train
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Ejecuta entrenamiento:

```bash
python train_patient_split.py ^
  --data-dir C:\Users\alexp\Downloads\redes neuronales\redes neuronales\mit-bih-arrhythmia-database-1.0.0 ^
  --artifacts-dir ..\artifacts ^
  --target-fs 250 ^
  --window-size 256 ^
  --pre-samples 96 ^
  --epochs 35
```

El script:

1. Lee registros WFDB con `wfdb.rdrecord`
2. Usa anotaciones `.atr` para mapear latidos a clases AAMI
3. Selecciona derivación automáticamente o la derivación indicada
4. Remuestrea, filtra y segmenta el ECG
5. Separa train/val/test por registro con `GroupShuffleSplit`
6. Entrena `Conv1D + BiLSTM`
7. Guarda `artifacts/model.keras` y `artifacts/metadata.json`

## Inferencia local

Backend:

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
set BACKEND_URL=http://127.0.0.1:8000
set PORT=8501
python app.py
```

## Docker

```bash
docker-compose up --build
```

## Endpoints principales

- `GET /health`
- `GET /model-info`
- `GET /model-architecture`
- `POST /analyze-record`
- `POST /predict-and-explain`
- `POST /pipeline-steps`

Los endpoints de inferencia reciben `multipart/form-data` con varios ficheros `files`.

## Tests

```bash
cd backend
python -m pytest tests -v
```

## Métricas objetivo

- Accuracy
- Macro-F1
- Balanced Accuracy
- Sensibilidad por clase
- Especificidad por clase
- Matriz de confusión

## Observaciones

- Si `artifacts/model.keras` no existe, el backend no hace inferencia y responderá con error 503.
- La explicabilidad usa gradientes sobre el segmento ECG representativo. Eso describe sensibilidad del modelo, no causalidad fisiológica absoluta.
