"""Test-only environment configuration."""

import os

os.environ.setdefault("TRIAGE_ML_API_KEY_SERVICE", "srv-" + "0" * 30)
os.environ.setdefault("TRIAGE_ML_API_KEY_DOCTOR", "doc-" + "0" * 30)
os.environ.setdefault("TRIAGE_ML_API_KEY_PATIENT", "pat-" + "0" * 30)
