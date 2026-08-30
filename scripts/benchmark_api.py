"""Benchmark baseline tool for Etapa 5 hook."""

import time
import json
import statistics
import platform
from pathlib import Path
from fastapi.testclient import TestClient

from triage_ml.api.app import create_app
from triage_ml.api.settings import Settings, get_settings


def run_benchmark() -> None:
    # 1. Configuração explícita
    # Não depender de chaves ambientais do desenvolvedor
    benchmark_settings = Settings(
        api_key_service="srv-" + "0" * 30,
        api_key_doctor="doc-" + "0" * 30,
        api_key_patient="pat-" + "0" * 30,
        log_level="ERROR",  # Reduz latência de I/O em tela para o teste
    )

    # 2. Instanciação e Injeção
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: benchmark_settings
    
    client = TestClient(app)
    headers = {"X-API-Key": benchmark_settings.api_key_doctor}
    payload = {
        "text": (
            "Patient presents with severe chest pain, elevated heart rate, "
            "and requires urgent cardiovascular checkup and continuous monitoring."
        )
    }

    warmup_requests = 50
    n_requests = 500
    latencies = []

    print(f"Iniciando warmup com {warmup_requests} requisições...")
    for _ in range(warmup_requests):
        resp = client.post("/predict", json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Warmup falhou: HTTP {resp.status_code} - {resp.text}")

    print(f"Iniciando benchmark com {n_requests} requisições...")
    for _ in range(n_requests):
        start = time.perf_counter()
        client.post("/predict", json=payload, headers=headers)
        latencies.append((time.perf_counter() - start) * 1000.0)

    # 3. Estatísticas e Extração de Métricas
    # statistics.quantiles (n=100) separa em 100 blocos. Índices: 49=p50, 94=p95, 98=p99.
    quantiles = statistics.quantiles(latencies, n=100)
    
    output = {
        "methodology": {
            "client": "fastapi.testclient.TestClient (internal ASGI routing, no network I/O)",
            "warmup_requests": warmup_requests,
            "measured_requests": n_requests,
            "payload_size_chars": len(payload["text"]),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "metrics": {
            "mean_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": quantiles[49],
            "p95_latency_ms": quantiles[94],
            "p99_latency_ms": quantiles[98],
        }
    }

    # 4. Gravação versionada para Etapa 5
    out_dir = Path("reports/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "api-prod-baseline.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    print(f"Benchmark concluído. Salvo em {out_path}")
    print(f" -> p50: {output['metrics']['p50_latency_ms']:.2f} ms")
    print(f" -> p95: {output['metrics']['p95_latency_ms']:.2f} ms")


if __name__ == "__main__":
    run_benchmark()