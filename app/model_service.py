"""Carga e inferencia del clasificador Keras."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.settings import get_settings

logger = logging.getLogger(__name__)

# El orden debe coincidir con train_ds.class_names del entrenamiento.
CLASS_NAMES = ["diseased", "healthy"]
CLASS_LABELS = {
    "diseased": "Hoja enferma",
    "healthy": "Hoja sana",
}


def resolve_model_path() -> Path | None:
    """Prioriza un archivo local; opcionalmente descarga desde Hugging Face Hub."""
    settings = get_settings()
    local_path = settings.model_path
    if local_path.is_file() and local_path.stat().st_size > 0:
        return local_path

    if not settings.hf_repo_id:
        return None

    try:
        # Importar el hub únicamente cuando se solicita una descarga remota.
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=settings.hf_model_filename,
            token=settings.hf_token,
        )
        return Path(downloaded)
    except Exception as exc:
        logger.exception("No fue posible descargar el modelo desde Hugging Face Hub: %s", exc)
        raise RuntimeError(
            "No fue posible descargar el modelo desde Hugging Face Hub. "
            "Verifica HF_REPO_ID, HF_MODEL_FILENAME y los permisos del token."
        ) from exc


def load_classifier() -> Any | None:
    """Carga el .keras. KerasHub registra las capas de preprocesamiento usadas."""
    model_path = resolve_model_path()
    if model_path is None:
        logger.warning(
            "Modelo no encontrado. Coloca el .keras en %s o configura HF_REPO_ID.",
            get_settings().model_path,
        )
        return None

    # El modelo del entrenamiento usa un preprocesador de KerasHub integrado
    # en el grafo. Importarlo antes de load_model registra sus clases serializadas.
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    try:
        import keras_hub  # noqa: F401  # registra componentes serializados
        import keras

        model = keras.saving.load_model(str(model_path), compile=False)

        # Comprobación defensiva del contrato definido durante el entrenamiento.
        input_shape = tuple(model.input_shape)
        output_shape = tuple(model.output_shape)
        expected_input = (None, get_settings().image_size, get_settings().image_size, 3)
        if input_shape != expected_input:
            raise ValueError(
                f"Entrada incompatible: {input_shape}; se esperaba {expected_input}."
            )
        if len(output_shape) != 2 or output_shape[-1] != len(CLASS_NAMES):
            raise ValueError(
                f"Salida incompatible: {output_shape}; se esperaba (None, 2)."
            )

        logger.info(
            "Modelo cargado desde %s; entrada=%s; salida=%s",
            model_path, input_shape, output_shape,
        )
        return model
    except Exception as exc:
        logger.exception("No se pudo cargar el modelo %s: %s", model_path, exc)
        raise RuntimeError(
            "No se pudo cargar el archivo .keras. Verifica que sea el clasificador "
            "ResNet50 entrenado y que las versiones de Keras/KerasHub coincidan "
            "con las usadas en Colab."
        ) from exc


def prepare_image(image_bytes: bytes, image_size: int) -> np.ndarray:
    """Convierte una imagen a RGB 224x224 y devuelve un batch float32 [0, 255].

    No se aplica rescale ni preprocess_input aquí: el preprocesamiento del
    entrenamiento está integrado en el modelo mediante modelo.preprocessor.
    """
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as opened:
        # image_dataset_from_directory(..., image_size=(224, 224)) usa
        # interpolación bilinear y ajusta la imagen a ese tamaño. Repetimos
        # la misma operación aquí para no cambiar el preprocesamiento previo.
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        # No normalizar aquí: el ResNetImageClassifierPreprocessor de KerasHub
        # está serializado dentro del .keras y realiza la conversión/escala.
        array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_probabilities(model: Any, batch: np.ndarray) -> list[float]:
    """Ejecuta inferencia y valida la salida softmax de dos clases."""
    output = model.predict(batch, verbose=0)
    probabilities = np.asarray(output, dtype=np.float64)

    if probabilities.ndim == 2 and probabilities.shape[0] == 1:
        probabilities = probabilities[0]
    if probabilities.shape != (len(CLASS_NAMES),):
        raise ValueError(
            f"Salida inesperada del modelo: {probabilities.shape}. Se esperaba (1, 2)."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("El modelo devolvió probabilidades no válidas.")

    # Pequeñas desviaciones por precisión numérica no cambian la clasificación.
    total = float(np.sum(probabilities))
    if total <= 0:
        raise ValueError("El modelo devolvió una suma de probabilidades no válida.")
    probabilities = probabilities / total
    return [float(value) for value in probabilities]
