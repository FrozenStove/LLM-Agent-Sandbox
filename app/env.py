from __future__ import annotations

from typing import TYPE_CHECKING

import logfire
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from fastapi import FastAPI


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    logfire_key: str | None = None
    llm_model: str = "gpt-4o"


settings = Settings()


def setup_observability(app: FastAPI) -> None:
    if settings.logfire_key:
        logfire.configure(token=settings.logfire_key)
        logfire.instrument_fastapi(app)
    else:
        print("Logfire key not found, observability will not be setup")
