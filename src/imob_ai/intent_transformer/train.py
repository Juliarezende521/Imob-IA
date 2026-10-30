"""Seleção em validação, congelamento e treino final local."""

import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.schemas import DatasetSplit
from imob_ai.intent_transformer.config import (
    ARTIFACT_ROOT,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    FINAL_MODEL_PATH,
    FROZEN_CONFIG_HASH_PATH,
    FROZEN_CONFIG_PATH,
    MAX_EPOCHS,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    RANDOM_SEED,
    SELECTION_PATH,
    SIMPLICITY_TIE_EPSILON,
    STRATEGIES,
    TRAINING_METADATA_PATH,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    Strategy,
    strategy_dict,
)
from imob_ai.intent_transformer.dataset import IntentTorchDataset
from imob_ai.intent_transformer.labels import ID2LABEL, LABEL2ID, taxonomy_hash
from imob_ai.intent_transformer.metrics import calculate_metrics, minimum_class_f1
from imob_ai.intent_transformer.model import (
    count_parameters,
    load_model,
    load_tokenizer,
)
from imob_ai.intent_transformer.tokenization import analyze_lengths


def set_seeds(seed: int = RANDOM_SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device() -> str:
    import torch

    requested = os.environ.get("IMOB_AI_DEVICE")
    if requested == "cpu":
        return "cpu"
    if requested not in {None, "mps"}:
        raise ValueError("IMOB_AI_DEVICE aceita somente cpu ou mps.")
    if torch.backends.mps.is_available():
        try:
            value = torch.tensor([1.0], device="mps")
            if (value + value).cpu().item() == 2.0:
                return "mps"
        except RuntimeError:
            pass
    return "cpu"


def split_hash(records: list[Any]) -> str:
    payload = "\n".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for item in records
    )
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dataloader(records: list[Any], tokenizer: Any, shuffle: bool) -> Any:
    from torch.utils.data import DataLoader

    generator = None
    if shuffle:
        import torch

        generator = torch.Generator().manual_seed(RANDOM_SEED)
    return DataLoader(
        IntentTorchDataset(records, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def _evaluate(model: Any, loader: Any, device: str) -> dict[str, Any]:
    import torch

    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
            expected.extend(batch["labels"].detach().cpu().tolist())
            predicted.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
    metrics = calculate_metrics(expected, predicted)
    metrics["loss"] = sum(losses) / len(losses)
    metrics["min_class_f1"] = minimum_class_f1(metrics)
    return metrics


def _train_strategy(
    strategy: Strategy,
    train_records: list[Any],
    validation_records: list[Any],
    tokenizer: Any,
    device: str,
) -> dict[str, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    set_seeds()
    model = load_model(strategy.freeze_encoder).to(device)
    parameters = count_parameters(model)
    train_loader = _dataloader(train_records, tokenizer, shuffle=True)
    validation_loader = _dataloader(validation_records, tokenizer, shuffle=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=strategy.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_key: Optional[tuple[float, float, float]] = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
        metrics = _evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, **metrics})
        key = (
            metrics["f1_macro"],
            metrics["min_class_f1"],
            metrics["recall_macro"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    duration = time.perf_counter() - started
    best_metrics = next(item for item in history if item["epoch"] == best_epoch)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return {
        "strategy": strategy_dict(strategy),
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "training_seconds": duration,
    }


def _choose_strategy(results: list[dict[str, Any]]) -> dict[str, Any]:
    frozen = next(x for x in results if x["strategy"]["freeze_encoder"])
    full = next(x for x in results if not x["strategy"]["freeze_encoder"])
    difference = abs(
        frozen["best_metrics"]["f1_macro"] - full["best_metrics"]["f1_macro"]
    )
    if difference <= SIMPLICITY_TIE_EPSILON:
        return frozen
    return max(
        results,
        key=lambda item: (
            item["best_metrics"]["f1_macro"],
            item["best_metrics"]["min_class_f1"],
            item["best_metrics"]["recall_macro"],
            item["strategy"]["freeze_encoder"],
            -item["training_seconds"],
        ),
    )


def _tokenizer_hash(tokenizer: Any) -> str:
    payload = json.dumps(
        tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _version(package: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(package)


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _frozen_config(
    selected: dict[str, Any],
    tokenizer: Any,
    lengths: dict[str, Any],
    device: str,
    splits: dict[str, list[Any]],
) -> dict[str, Any]:
    from imob_ai.intent.challenge import CHALLENGE_PATH, sha256_file

    return {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_hash": _tokenizer_hash(tokenizer),
        "label2id": LABEL2ID,
        "id2label": {str(key): value for key, value in ID2LABEL.items()},
        "taxonomy_hash": taxonomy_hash(),
        "dataset_hashes": {
            name: split_hash(records) for name, records in splits.items()
        }
        | {"challenge": sha256_file(CHALLENGE_PATH)},
        "strategy": selected["strategy"],
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "selected_epochs": selected["best_epoch"],
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": RANDOM_SEED,
        "validation_metrics": selected["best_metrics"],
        "length_analysis": lengths,
        "device": device,
        "versions": {
            "python": platform.python_version(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "sentence_transformers": _version("sentence-transformers"),
        },
        "platform": {"machine": platform.machine(), "system": platform.platform()},
        "determinism_note": (
            "Seeds fixadas; determinismo absoluto pode variar entre dispositivos "
            "e operações."
        ),
    }


def _train_final(
    selected: dict[str, Any], records: list[Any], tokenizer: Any, device: str
) -> dict[str, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    set_seeds()
    strategy = Strategy(**selected["strategy"])
    model = load_model(strategy.freeze_encoder).to(device)
    parameters = count_parameters(model)
    loader = _dataloader(records, tokenizer, shuffle=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=strategy.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(loader) * selected["best_epoch"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    started = time.perf_counter()
    model.train()
    for _ in range(selected["best_epoch"]):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
    duration = time.perf_counter() - started
    FINAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_PATH, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MODEL_PATH)
    return {
        "training_seconds": duration,
        "parameters": parameters,
        "artifact_sha256": directory_sha256(FINAL_MODEL_PATH),
        "artifact_bytes": sum(
            path.stat().st_size
            for path in FINAL_MODEL_PATH.rglob("*")
            if path.is_file()
        ),
        "device": device,
    }


def main() -> None:
    if FROZEN_CONFIG_PATH.exists() or FINAL_MODEL_PATH.exists():
        raise RuntimeError(
            "Configuração ou artefato já existe; recuse novo treinamento."
        )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    records = load_intent_dataset()
    splits = {
        split.value: [item for item in records if item.split == split]
        for split in DatasetSplit
    }
    tokenizer = load_tokenizer()
    development = [*splits["train"], *splits["validation"]]
    lengths = asdict(analyze_lengths(tokenizer, development, MAX_LENGTH))
    device = choose_device()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [
        _train_strategy(
            strategy,
            splits["train"],
            splits["validation"],
            tokenizer,
            device,
        )
        for strategy in STRATEGIES
    ]
    selected = _choose_strategy(results)
    selection = {"candidates": results, "selected": selected["strategy"]["name"]}
    _write_json(SELECTION_PATH, selection)

    frozen = _frozen_config(selected, tokenizer, lengths, device, splits)
    _write_json(FROZEN_CONFIG_PATH, frozen)
    config_hash = file_sha256(FROZEN_CONFIG_PATH)
    FROZEN_CONFIG_HASH_PATH.write_text(config_hash + "\n", encoding="utf-8")

    final_metadata = _train_final(selected, development, tokenizer, device)
    final_metadata["frozen_config_sha256"] = config_hash
    _write_json(TRAINING_METADATA_PATH, final_metadata)
    print(
        json.dumps(
            {"selection": selection, "frozen": frozen, "final": final_metadata},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
