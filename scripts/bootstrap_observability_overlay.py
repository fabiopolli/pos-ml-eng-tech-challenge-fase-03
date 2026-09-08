"""Bootstrap the observability overlay (infra/docker-compose.yml).

Writes a fully-valid ``.env`` file in the project root, with the 4
variables required by ``infra/docker-compose.yml``:

  * ``MODEL_VERSION`` — version timestamp of the model artefact
    (``YYYYMMDDTHHMMSSZ-<12hex>``). Auto-detected from
    ``models/<latest>/`` when available; otherwise the caller is
    asked to provide one (or run ``uv run triage-ml-train`` first).
  * ``TRIAGE_ML_API_KEY_{SERVICE,DOCTOR,PATIENT}`` — HMAC-SHA-256
    fingerprints for the three roles. Generated with
    ``secrets.token_urlsafe(32)`` when missing.
  * ``GRAFANA_ADMIN_PASSWORD`` — admin password for Grafana.
    Generated with ``secrets.token_urlsafe(32)`` when missing.

Run::

    uv run python scripts/bootstrap_observability_overlay.py

The script is idempotent: existing keys are preserved unless the
caller passes ``--force``.

This avoids the ``cat > .env <<EOF ... EOF`` foot-gun documented in
the Etapa 5/6 review cycle (heredocs on bare shells can be parsed
as commands; ``docker compose`` interpolates env vars only when
they are present *before* invocation).
"""

from __future__ import annotations

import argparse
import secrets
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
MODELS_DIR = REPO_ROOT / "models"
PLACEHOLDER = "PLACEHOLDER"


def _existing_env() -> dict[str, str]:
    if not ENV_FILE.is_file():
        return {}
    parsed: dict[str, str] = {}
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parsed[key.strip()] = value.strip()
    return parsed


def _detect_latest_model_version() -> str | None:
    if not MODELS_DIR.is_dir():
        return None
    candidates: list[tuple[str, Path]] = []
    for entry in MODELS_DIR.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        # Accepted shape: YYYYMMDDTHHMMSSZ-<12hex> (28 chars total).
        if len(name) != 29 or name[16] != "-":
            continue
        timestamp = name[:16]
        try:
            datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
        except ValueError:
            continue
        candidates.append((name, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][0]


def _ensure_key(env: dict[str, str], key: str, *, force: bool) -> str:
    current = env.get(key, "")
    placeholder = current.startswith(PLACEHOLDER) or current.startswith("REPLACE")
    if current and not placeholder and not force:
        return current
    return secrets.token_urlsafe(32)


def _ensure_model_version(env: dict[str, str], *, force: bool, override: str | None) -> str:
    if override:
        return override
    current = env.get("MODEL_VERSION", "")
    if (
        current
        and not current.startswith(PLACEHOLDER)
        and not current.startswith("REPLACE")
        and not force
    ):
        return current
    detected = _detect_latest_model_version()
    if detected:
        return detected
    return "REPLACE_WITH_uv_run_triage-ml-train_OUTPUT"


def build_env(*, force: bool = False, model_version: str | None = None) -> dict[str, str]:
    env = _existing_env()

    env["MODEL_VERSION"] = _ensure_model_version(env, force=force, override=model_version)
    env["TRIAGE_ML_API_KEY_SERVICE"] = _ensure_key(env, "TRIAGE_ML_API_KEY_SERVICE", force=force)
    env["TRIAGE_ML_API_KEY_DOCTOR"] = _ensure_key(env, "TRIAGE_ML_API_KEY_DOCTOR", force=force)
    env["TRIAGE_ML_API_KEY_PATIENT"] = _ensure_key(env, "TRIAGE_ML_API_KEY_PATIENT", force=force)
    env["GRAFANA_ADMIN_PASSWORD"] = _ensure_key(env, "GRAFANA_ADMIN_PASSWORD", force=force)

    env.setdefault("GRAFANA_ADMIN_USER", "admin")
    env.setdefault("TRIAGE_ML_LOG_LEVEL", "INFO")
    env.setdefault("TRIAGE_ML_RATELIMIT_DEFAULT", "120/minute")
    env.setdefault("TRIAGE_ML_RATELIMIT_PREDICT", "120/minute")
    return env


def render_env(env: dict[str, str]) -> str:
    lines = [
        "# ============================================================================",
        "# Tech Challenge Fase 3 — observability overlay (.env).",
        "# Generated/refreshed by scripts/bootstrap_observability_overlay.py.",
        "#",
        "# Required by infra/docker-compose.yml (`docker compose` fails fast if any",
        "# of MODEL_VERSION / TRIAGE_ML_API_KEY_* / GRAFANA_ADMIN_PASSWORD is empty).",
        "# ============================================================================",
        "",
        "# Version timestamp of the model artefact under models/.",
        "# Train a new one with: uv run triage-ml-train",
        f"MODEL_VERSION={env['MODEL_VERSION']}",
        "",
        "# Three API keys (HMAC-SHA-256 fingerprints). Regenerate with:",
        '#   python -c "import secrets; print(secrets.token_urlsafe(32))"',
        f"TRIAGE_ML_API_KEY_SERVICE={env['TRIAGE_ML_API_KEY_SERVICE']}",
        f"TRIAGE_ML_API_KEY_DOCTOR={env['TRIAGE_ML_API_KEY_DOCTOR']}",
        f"TRIAGE_ML_API_KEY_PATIENT={env['TRIAGE_ML_API_KEY_PATIENT']}",
        "",
        "# Grafana admin password.",
        f"GRAFANA_ADMIN_USER={env.get('GRAFANA_ADMIN_USER', 'admin')}",
        f"GRAFANA_ADMIN_PASSWORD={env['GRAFANA_ADMIN_PASSWORD']}",
        "",
        "# Optional tunables (defaults shown).",
        f"TRIAGE_ML_LOG_LEVEL={env.get('TRIAGE_ML_LOG_LEVEL', 'INFO')}",
        f"TRIAGE_ML_RATELIMIT_DEFAULT={env.get('TRIAGE_ML_RATELIMIT_DEFAULT', '120/minute')}",
        f"TRIAGE_ML_RATELIMIT_PREDICT={env.get('TRIAGE_ML_RATELIMIT_PREDICT', '120/minute')}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate secrets even when .env already has valid keys",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Override MODEL_VERSION (default: latest under models/)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the rendered .env to stdout instead of writing the file",
    )
    args = parser.parse_args()

    env = build_env(force=args.force, model_version=args.model_version)
    rendered = render_env(env)

    if args.print_only:
        sys.stdout.write(rendered)
        return 0

    ENV_FILE.write_text(rendered, encoding="utf-8")
    print(f"wrote {ENV_FILE}")
    print(f"MODEL_VERSION={env['MODEL_VERSION']}")
    if not (MODELS_DIR / env["MODEL_VERSION"]).is_dir():
        print(
            "  ⚠ MODEL_VERSION directory not found under models/. Train a "
            "model with `uv run triage-ml-train` before bringing up the overlay.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
