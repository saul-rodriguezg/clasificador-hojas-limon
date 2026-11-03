"""API REST y frontend para clasificar hojas de limón con ResNet50."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from app.model_service import (
    CLASS_LABELS,
    CLASS_NAMES,
    load_classifier,
    predict_probabilities,
    prepare_image,
)
from app.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una vez al iniciar el proceso."""
    try:
        app.state.model = load_classifier()
        app.state.model_error = None
    except Exception as exc:
        app.state.model = None
        app.state.model_error = str(exc)
        logger.exception("La API inició sin modelo debido a un error de carga.")
    yield


app = FastAPI(
    title="Clasificador de hojas de limón",
    description=(
        "API para clasificar fotografías de hojas de limón como enfermas "
        "(diseased) o sanas (healthy), usando un modelo ResNet50."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def home():
    with open("app/static/index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.get("/health", tags=["Monitoreo"])
async def health():
    """Indica si el proceso está activo y si el modelo está cargado."""
    model_loaded = getattr(app.state, "model", None) is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_error": getattr(app.state, "model_error", None),
        "classes": CLASS_NAMES,
    }


@app.post("/predict", tags=["Predicción"])
async def predict(file: UploadFile = File(...)):
    """Recibe una imagen y devuelve la clase predicha y sus probabilidades."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo todavía no está disponible. Coloca el archivo .keras "
                "en la ruta configurada o configura HF_REPO_ID y reinicia la API."
            ),
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El archivo debe ser una imagen.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    image_bytes = await file.read(max_bytes + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"La imagen supera el límite de {settings.max_upload_mb} MB.",
        )

    try:
        batch = prepare_image(image_bytes, settings.image_size)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen. Verifica el archivo.")

    try:
        probabilities = predict_probabilities(model, batch)
    except ValueError as exc:
        logger.exception("Salida inválida durante la predicción.")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Error durante la predicción.")
        raise HTTPException(status_code=500, detail="Ocurrió un error al ejecutar el modelo.") from exc

    predicted_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    predicted_class = CLASS_NAMES[predicted_index]
    return {
        "prediction": predicted_class,
        "class": CLASS_LABELS[predicted_class],
        "confidence": round(probabilities[predicted_index], 6),
        "probabilities": {
            CLASS_NAMES[index]: round(probabilities[index], 6)
            for index in range(len(CLASS_NAMES))
        },
    }
