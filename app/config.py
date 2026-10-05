from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./local.db"
    llm_provider: str = "anthropic"  # anthropic | openai | none
    llm_model: str = "claude-sonnet-5-5"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment: str = "agentic-data-scientist"
    data_dir: str = "./data"
    max_upload_mb: int = 50
    max_train_rows: int = 50_000


settings = Settings()
