"""Configuración de la aplicación mediante variables de entorno."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    hf_repo_id: str | None
    hf_model_filename: str
    hf_token: str | None
    image_size: int
    max_upload_mb: int
    port: int


def get_settings() -> Settings:
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", "models/ResNet50.keras")),
        hf_repo_id=os.getenv("HF_REPO_ID") or None,
        hf_model_filename=os.getenv("HF_MODEL_FILENAME", "ResNet50.keras"),
        hf_token=os.getenv("HF_TOKEN") or None,
        image_size=int(os.getenv("IMAGE_SIZE", "224")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "10")),
        port=int(os.getenv("PORT", "8000")),
    )
