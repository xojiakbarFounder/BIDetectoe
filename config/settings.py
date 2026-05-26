"""
Central application settings loaded from environment / .env file.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


def resolve_torch_device(device: str) -> str:
    """Resolve 'auto' to CUDA when available, otherwise CPU."""
    requested = (device or "auto").strip().lower()
    if requested != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "surveillance"
    postgres_user: str = "surveillance_user"
    postgres_password: str = "change_me_secret"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode=require"
        )

    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?ssl=require"
        )

    # ── Stream ────────────────────────────────────────────────────────────────
    youtube_stream_url: str = "https://www.youtube.com/watch?v=5qap5aO4i9A"
    target_fps: int = 10

    # ── YOLO ──────────────────────────────────────────────────────────────────
    yolo_model: str = "yolov8n.pt"
    yolo_pretrained_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.40
    yolo_device: str = "auto"
    yolo_person_class_id: int = 0

    @property
    def resolved_yolo_device(self) -> str:
        return resolve_torch_device(self.yolo_device)

    # Training / fine-tuning
    dataset_yaml: str = "dataset/dataset.yaml"
    training_project: str = "experiments"
    training_name: str = "surveillance-yolov8"
    trained_models_dir: str = "models"
    train_epochs: int = 50
    train_batch: int = 16
    train_imgsz: int = 640

    # ── Virtual counting line (normalised 0–1) ────────────────────────────────
    line_start_x: float = 0.0
    line_start_y: float = 0.5
    line_end_x: float = 1.0
    line_end_y: float = 0.5

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # ── Timezone ──────────────────────────────────────────────────────────────
    utc_offset_hours: int = 5          # UTC+5 = Toshkent

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"


# Singleton used throughout the application
settings = Settings()
