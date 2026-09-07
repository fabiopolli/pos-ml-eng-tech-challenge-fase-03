#!/usr/bin/env bash
# Pre-flight validation for the Airflow standalone container.
# Fails fast when TRIAGE_REQUIRE_AUTH=true but DagsHub credentials are missing.
set -euo pipefail

if [ "${TRIAGE_REQUIRE_AUTH:-false}" = "true" ]; then
    if [ -z "${DAGSHUB_USERNAME:-}" ] || [ -z "${DAGSHUB_USER_TOKEN:-}" ]; then
        echo "[entrypoint] TRIAGE_REQUIRE_AUTH=true requires DAGSHUB_USERNAME and DAGSHUB_USER_TOKEN" >&2
        exit 1
    fi
fi

exec "$@"