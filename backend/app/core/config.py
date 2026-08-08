from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Sahachaara"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/sahachaara"

    # Security
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # CORS
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Gemini AI & Debug Settings
    gemini_api_key: str = ""
    gemini_primary_model: str = "gemini-3.5-flash"
    gemini_fallback_models: str = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    debug_ai: bool = True
    demo_mode: bool = False
    next_public_api_url: str = "http://localhost:8000"


settings = Settings()


