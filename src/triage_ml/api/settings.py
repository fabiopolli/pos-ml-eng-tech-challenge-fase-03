"""Environment settings and strict constraints for production API."""
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_ML_", env_file=".env", extra="ignore")

    api_key_service: str = Field(min_length=32)
    api_key_doctor: str = Field(min_length=32)
    api_key_patient: str = Field(min_length=32)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    ratelimit_default: str = "60/minute"
    ratelimit_predict: str = "30/minute"

from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()