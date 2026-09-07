"""Environment settings and strict constraints for production API."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Compose may read ``.env`` and passes only this service's declared variables.
    # The application itself reads process environment only, preventing unrelated
    # secrets from other services from being included in validation errors.
    model_config = SettingsConfigDict(env_prefix="TRIAGE_ML_", extra="forbid")

    api_key_service: str = Field(min_length=32)
    api_key_doctor: str = Field(min_length=32)
    api_key_patient: str = Field(min_length=32)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    ratelimit_default: str = "60/minute"
    ratelimit_predict: str = "30/minute"

    @model_validator(mode="after")
    def _keys_must_be_distinct(self) -> "Settings":
        keys = {
            "api_key_service": self.api_key_service,
            "api_key_doctor": self.api_key_doctor,
            "api_key_patient": self.api_key_patient,
        }
        duplicates = {
            name: value for name, value in keys.items() if list(keys.values()).count(value) > 1
        }
        if duplicates:
            raise ValueError(
                "api_key_service, api_key_doctor and api_key_patient must be distinct; "
                "reusing the same value across roles would silently collapse role checks"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
