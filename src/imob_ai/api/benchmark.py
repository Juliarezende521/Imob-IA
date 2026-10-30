"""Benchmark básico e local da API com o baseline vencedor."""

import json
import platform
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from imob_ai.api.app import create_app

OUTPUT_PATH = Path("docs/experiments/api-benchmark.json")
REPETITIONS = 30


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "median_ms": statistics.median(ordered) * 1000,
        "p95_ms": ordered[p95_index] * 1000,
        "repetitions": len(values),
    }


def measure(operation) -> list[float]:
    values = []
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        response = operation()
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            raise RuntimeError(f"Benchmark recebeu HTTP {response.status_code}.")
        values.append(elapsed)
    return values


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("Benchmark da API já existe.")
    client = TestClient(create_app())
    startup_started = time.perf_counter()
    with client:
        startup_seconds = time.perf_counter() - startup_started

        def classify():
            return client.post(
                "/v1/intents/classify",
                json={"message": "Quero conhecer o apartamento sintético amanhã"},
            )

        def search():
            return client.post(
                "/v1/properties/search",
                json={"text": "casa com quintal", "city": "Franca", "limit": 5},
            )

        def route():
            return client.post(
                "/v1/assistant/route",
                json={
                    "message": "Procuro uma casa sintética para alugar em Franca",
                    "search_filters": {"city": "Franca", "purpose": "aluguel"},
                },
            )

        classify()
        search()
        route()
        result = {
            "startup_ms": startup_seconds * 1000,
            "warmup_requests_per_endpoint": 1,
            "classification": summarize(measure(classify)),
            "property_search": summarize(measure(search)),
            "assistant_route": summarize(measure(route)),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "scope_note": "Benchmark local; não representa capacidade de produção.",
        }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
