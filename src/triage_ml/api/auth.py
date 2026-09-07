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
    """Dependency that ensures the authenticated role is within allowed roles.

    The ``allowed_roles`` tuple is captured at instantiation time so callers
    cannot mutate it after the dependency has been mounted on a route.
    """

    def __init__(self, allowed_roles: list[str] | tuple[str, ...]):
        self.allowed_roles: frozenset[str] = frozenset(allowed_roles)

    def __call__(self, role: str = Depends(get_current_role)) -> str:
        if role not in self.allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
        return role


class RequirePredictRole:
    """Dependency for ``/predict`` that preserves the patient-specific error code.

    Centralising this here keeps the RBAC contract single-sourced: a future
    new role that reaches ``/predict`` still goes through the same gate, but
    the patient role is reported with ``clinician_review_required`` so
    dashboards and clients can distinguish "no clinical predictions for
    patient sessions" from "API key has no permission".
    """

    def __call__(self, role: str = Depends(get_current_role)) -> str:
        if role == "patient":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="clinician_review_required"
            )
        if role != "doctor":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
        return role
