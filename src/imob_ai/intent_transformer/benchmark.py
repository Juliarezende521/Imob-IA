"""Benchmark local repetido sem acessar o challenge."""

import json
import pickle
import statistics
import time
from typing import Callable

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.features import CANDIDATE_CONFIGS
from imob_ai.intent.model import build_pipeline
from imob_ai.intent.schemas import DatasetSplit
from imob_ai.intent.train import SELECTED_CONFIG_NAME
from imob_ai.intent_transformer.config import BENCHMARK_PATH
from imob_ai.intent_transformer.runtime import (
    load_and_verify_frozen,
    load_local_transformer,
)
from imob_ai.intent_transformer.train import choose_device


def summarize_timings(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "median_ms": 1000 * statistics.median(ordered),
        "p95_ms": 1000 * ordered[p95_index],
        "repetitions": len(ordered),
    }


def measure(operation: Callable[[], None], repetitions: int) -> list[float]:
    values = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        values.append(time.perf_counter() - started)
    return values


def main() -> None:
    import torch

    if BENCHMARK_PATH.exists():
        raise RuntimeError("Benchmark já registrado.")
    config, metadata = load_and_verify_frozen()
    records = load_intent_dataset()
    train_validation = [
        item
        for item in records
        if item.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
    ]
    validation = [item for item in records if item.split == DatasetSplit.VALIDATION]
    texts = [item.text for item in validation]

    feature_config = next(
        item for item in CANDIDATE_CONFIGS if item.name == SELECTED_CONFIG_NAME
    )
    baseline_started = time.perf_counter()
    baseline = build_pipeline(feature_config)
    baseline.fit(
        [item.text for item in train_validation],
        [item.intent.value for item in train_validation],
    )
    baseline_training_seconds = time.perf_counter() - baseline_started
    baseline.predict([texts[0]])
    baseline_single = measure(lambda: baseline.predict([texts[0]]), 30)
    baseline_batch = measure(lambda: baseline.predict(texts[:8]), 20)

    device = choose_device()
    tokenizer, model = load_local_transformer(device)

    def synchronize() -> None:
        if device == "mps":
            torch.mps.synchronize()

    def infer(batch: list[str]) -> None:
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded)
        synchronize()

    infer([texts[0]])
    infer(texts[:8])
    transformer_single = measure(lambda: infer([texts[0]]), 30)
    transformer_batch = measure(lambda: infer(texts[:8]), 20)

    load_values = []
    for _ in range(3):
        started = time.perf_counter()
        loaded_tokenizer, loaded_model = load_local_transformer(device)
        synchronize()
        load_values.append(time.perf_counter() - started)
        del loaded_tokenizer, loaded_model
        if device == "mps":
            torch.mps.empty_cache()

    result = {
        "device": device,
        "warmup": {"single": 1, "batch_8": 1},
        "baseline": {
            "training_seconds": baseline_training_seconds,
            "serialized_bytes": len(pickle.dumps(baseline)),
            "single": summarize_timings(baseline_single),
            "batch_8": summarize_timings(baseline_batch),
            "batch_throughput_messages_per_second": (
                8 / statistics.median(baseline_batch)
            ),
        },
        "transformer": {
            "training_seconds": metadata["training_seconds"],
            "artifact_bytes": metadata["artifact_bytes"],
            "load": summarize_timings(load_values),
            "single": summarize_timings(transformer_single),
            "batch_8": summarize_timings(transformer_batch),
            "batch_throughput_messages_per_second": (
                8 / statistics.median(transformer_batch)
            ),
        },
        "memory_note": "RAM não medida com precisão neste ambiente.",
    }
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
