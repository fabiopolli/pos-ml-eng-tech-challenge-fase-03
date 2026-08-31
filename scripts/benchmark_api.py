"""Benchmark baseline tool for Etapa 5 hook.

Este script mede a latência da API FastAPI de produção rodando
via rede (localhost ou cloud). Requer que a API esteja ativa e
que a variável TRIAGE_ML_API_KEY_DOCTOR esteja populada no ambiente.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import requests


def run_benchmark() -> None:
    parser = argparse.ArgumentParser(description="Benchmark da API de triagem em produção.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="URL base da API em execução (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    api_key = os.environ.get("TRIAGE_ML_API_KEY_DOCTOR")
    if not api_key:
        print(
            "ERRO: a variável de ambiente TRIAGE_ML_API_KEY_DOCTOR não está definida.",
            file=sys.stderr,
        )
        sys.exit(1)

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    # Payload com texto sintético. Suficiente para passar pelo langid local
    # sem usar nenhum laudo real ou expor dados sensíveis.
    payload = {
        "text": (
            "Patient presents with severe chest pain, elevated heart rate, "
            "and requires urgent cardiovascular checkup and continuous monitoring."
        )
    }

    print(f"Conectando a {base_url} para recuperar informações do modelo...")
    try:
        # A nova rota model-info requer a key do doctor ou service
        info_resp = requests.get(f"{base_url}/model-info", headers=headers, timeout=10.0)
        info_resp.raise_for_status()
        model_info = info_resp.json()

        model_version = model_info.get("model_version", "unknown")
        dependency_versions = model_info.get("dependency_versions", {})
    except requests.RequestException as exc:
        print(f"ERRO ao acessar /model-info: {exc}", file=sys.stderr)
        sys.exit(1)

    warmup_requests = 50
    n_requests = 500
    latencies = []
    predict_url = f"{base_url}/predict"

    print(f"Iniciando warmup com {warmup_requests} requisições...")
    for _ in range(warmup_requests):
        resp = requests.post(predict_url, json=payload, headers=headers, timeout=10.0)
        if resp.status_code != 200:
            print(f"ERRO: Warmup falhou com HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
            sys.exit(1)

    print(f"Iniciando benchmark com {n_requests} requisições...")
    for _ in range(n_requests):
        start = time.perf_counter()
        resp = requests.post(predict_url, json=payload, headers=headers, timeout=10.0)
        latency = (time.perf_counter() - start) * 1000.0

        if resp.status_code != 200:
            print(f"ERRO: Medição falhou com HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
            sys.exit(1)

        latencies.append(latency)

    # statistics.quantiles com n=100 gera 99 cortes.
    # Índices: [49] = p50, [94] = p95, [98] = p99
    quantiles = statistics.quantiles(latencies, n=100)

    output = {
        "methodology": {
            "client": "requests (HTTP via rede)",
            "warmup_requests": warmup_requests,
            "measured_requests": n_requests,
            "payload_size_chars": len(payload["text"]),
            "base_url": base_url,
            "rate_limit_predict": os.environ.get("TRIAGE_ML_RATELIMIT_PREDICT"),
        },
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "dependency_versions": dependency_versions,
        },
        "model": {
            "model_version": model_version,
        },
        "metrics": {
            "mean_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": quantiles[49],
            "p95_latency_ms": quantiles[94],
            "p99_latency_ms": quantiles[98],
        },
    }

    out_dir = Path("reports/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "api-prod-baseline.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nBenchmark concluído. Salvo em {out_path}")
    print(f" -> Média: {output['metrics']['mean_latency_ms']:.2f} ms")
    print(f" -> p50:   {output['metrics']['p50_latency_ms']:.2f} ms")
    print(f" -> p95:   {output['metrics']['p95_latency_ms']:.2f} ms")
    print(f" -> p99:   {output['metrics']['p99_latency_ms']:.2f} ms")


if __name__ == "__main__":
    run_benchmark()
