"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_")

    model_path: str = "artifacts/model.joblib"
    model_version: str = "1.0.0"
    log_level: str = "INFO"
    workers: int = 1
    enable_metrics: bool = True


settings = Settings()
