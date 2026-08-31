"""Role-Based Access Control and authentication dependencies."""

import hmac

from fastapi import Depends, Header, HTTPException, status

from triage_ml.api.settings import Settings, get_settings


def get_current_role(
    settings: Settings = Depends(get_settings),  # noqa: B008
    api_key: str = Header(alias="X-API-Key", default=""),
) -> str:
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")

    encoded_key = api_key.encode("utf-8")
    if hmac.compare_digest(encoded_key, settings.api_key_service.encode("utf-8")):
        return "service"
    if hmac.compare_digest(encoded_key, settings.api_key_doctor.encode("utf-8")):
        return "doctor"
    if hmac.compare_digest(encoded_key, settings.api_key_patient.encode("utf-8")):
        return "patient"

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")


class RequireRole:
    """Dependency that ensures the authenticated role is within allowed roles."""

    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, role: str = Depends(get_current_role)) -> str:
        if role not in self.allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
        return role
