"""Reconstrução e exportação segura do baseline aprovado."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import sklearn

from imob_ai.hub.metadata import (
    FROZEN_METRICS,
    GITHUB_URL,
    LABELS,
    LICENSE_STATUS,
    MODEL_REQUIREMENTS,
    PACKAGE_VERSION,
    PROJECT_ROOT,
    file_sha256,
    write_manifest,
)
from imob_ai.intent.dataset import load_intent_dataset
from imob_ai.intent.features import CANDIDATE_CONFIGS
from imob_ai.intent.model import build_pipeline
from imob_ai.intent.schemas import DatasetSplit
from imob_ai.intent.train import SELECTED_CONFIG_NAME

EXAMPLE_MESSAGES = [
    "Quero alugar uma casa em Franca",
    "Preciso falar com alguém sobre o boleto",
    "O chuveiro do imóvel parou de funcionar",
]


def _normalize_skops_archive(model_path: Path) -> None:
    """Remove runtime object ids and ZIP timestamps from a skops archive."""
    extracted: dict[str, bytes] = {}
    with zipfile.ZipFile(model_path, "r") as archive:
        for name in archive.namelist():
            extracted[name] = archive.read(name)

    schema = json.loads(extracted.pop("schema.json").decode("utf-8"))
    id_map: dict[int, int] = {}

    def normalize_ids(value: Any) -> None:
        if isinstance(value, dict):
            object_id = value.get("__id__")
            if isinstance(object_id, int):
                if object_id not in id_map:
                    id_map[object_id] = len(id_map) + 1
                value["__id__"] = id_map[object_id]
            for child in value.values():
                normalize_ids(child)
            stop_words_id = value.get("_stop_words_id")
            if isinstance(stop_words_id, dict) and "content" in stop_words_id:
                stop_words_id["content"] = "0"
        elif isinstance(value, list):
            for child in value:
                normalize_ids(child)

    normalize_ids(schema)

    renamed: dict[str, bytes] = {}
    filename_map: dict[str, str] = {}
    for index, (name, content) in enumerate(
        sorted(
            extracted.items(),
            key=lambda item: (hashlib.sha256(item[1]).hexdigest(), item[0]),
        ),
        start=1,
    ):
        suffix = Path(name).suffix
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        stable_name = f"array-{index:02d}-{content_hash}{suffix}"
        filename_map[name] = stable_name
        renamed[stable_name] = content

    def normalize_filenames(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "file" and isinstance(child, str) and child in filename_map:
                    value[key] = filename_map[child]
                else:
                    normalize_filenames(child)
        elif isinstance(value, list):
            for child in value:
                normalize_filenames(child)

    normalize_filenames(schema)
    renamed["schema.json"] = (
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")

    normalized_path = model_path.with_suffix(".normalized")
    with zipfile.ZipFile(
        normalized_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in sorted(renamed):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, renamed[name])
    normalized_path.replace(model_path)


def build_final_pipeline():
    records = load_intent_dataset()
    development = [
        record
        for record in records
        if record.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
    ]
    if len(development) != 320:
        raise ValueError("O baseline final exige exatamente train + validation.")
    config = next(
        item for item in CANDIDATE_CONFIGS if item.name == SELECTED_CONFIG_NAME
    )
    if (
        config.name != "word_char"
        or config.ngram_range != (1, 2)
        or config.c != 1.0
        or config.class_weight is not None
    ):
        raise ValueError("Configuração final diverge do baseline congelado.")
    pipeline = build_pipeline(config)
    pipeline.fit(
        [record.text for record in development],
        [record.intent.value for record in development],
    )
    return pipeline


def _prediction_payload(pipeline: Any) -> list[dict[str, object]]:
    probabilities = pipeline.predict_proba(EXAMPLE_MESSAGES)
    classes = pipeline.named_steps["classifier"].classes_
    predictions = pipeline.predict(EXAMPLE_MESSAGES)
    output = []
    for message, predicted, scores in zip(EXAMPLE_MESSAGES, predictions, probabilities):
        ranked = sorted(zip(classes, scores), key=lambda item: -float(item[1]))
        output.append(
            {
                "input": message,
                "intent": str(predicted),
                "model_score": float(ranked[0][1]),
                "score_notice": "Relative model score; not a calibrated probability.",
                "human_review": "Apply the separately documented operational policy.",
            }
        )
    return output


def _safe_round_trip(model_path: Path, pipeline: Any) -> list[str]:
    from skops.io import get_untrusted_types, load

    untrusted = get_untrusted_types(file=model_path)
    if untrusted:
        raise ValueError(f"Tipos não confiáveis no artefato: {untrusted}")
    restored = load(model_path, trusted=[])
    before = pipeline.predict(EXAMPLE_MESSAGES).tolist()
    after = restored.predict(EXAMPLE_MESSAGES).tolist()
    if before != after:
        raise ValueError("Predições divergiram após round trip skops.")
    if not np.allclose(
        pipeline.predict_proba(EXAMPLE_MESSAGES),
        restored.predict_proba(EXAMPLE_MESSAGES),
        rtol=0,
        atol=0,
    ):
        raise ValueError("Scores divergiram após round trip skops.")
    return untrusted


def export_model(package_dir: Path, dataset_hash: str) -> dict[str, object]:
    from skops.io import dump

    package_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_final_pipeline()
    model_path = package_dir / "model.skops"
    dump(pipeline, model_path)
    _normalize_skops_archive(model_path)
    untrusted = _safe_round_trip(model_path, pipeline)
    classifier_classes = [
        str(label) for label in pipeline.named_steps["classifier"].classes_
    ]
    labels = {
        "schema_version": "1",
        "canonical_order": LABELS,
        "classifier_classes": classifier_classes,
    }
    (package_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "schema_version": "1",
        "model_file": "model.skops",
        "task": "text-classification",
        "library_name": "skops",
        "architecture": "word_char_tfidf_logistic_regression",
        "input": {"type": "string", "field": "text", "non_empty": True},
        "output": {"label_field": "intent", "score_field": "model_score"},
        "score_calibrated": False,
        "training_splits": ["train", "validation"],
        "training_records": 320,
        "hyperparameters": {
            "word_ngram_range": [1, 2],
            "char_analyzer": "char_wb",
            "char_ngram_range": [3, 5],
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "seed": 42,
        },
    }
    (package_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (package_dir / "example_input.json").write_text(
        json.dumps({"inputs": EXAMPLE_MESSAGES}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (package_dir / "example_output.json").write_text(
        json.dumps(
            {"predictions": _prediction_payload(pipeline)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    requirements = "\n".join(
        [
            f"numpy=={MODEL_REQUIREMENTS['numpy']}",
            f"scikit-learn=={MODEL_REQUIREMENTS['scikit-learn']}",
            f"skops=={MODEL_REQUIREMENTS['skops']}",
        ]
    )
    (package_dir / "requirements.txt").write_text(requirements + "\n", encoding="utf-8")
    model_hash = file_sha256(model_path)
    template = Template(
        (PROJECT_ROOT / "hub" / "model-card-template.md").read_text(encoding="utf-8")
    )
    label_list = "\n".join(f"- `{label}`" for label in LABELS)
    (package_dir / "README.md").write_text(
        template.substitute(
            version=PACKAGE_VERSION,
            model_hash=model_hash,
            dataset_hash=dataset_hash,
            github_url=GITHUB_URL,
            license_status=LICENSE_STATUS,
            labels=label_list,
        ),
        encoding="utf-8",
    )
    manifest = write_manifest(
        package_dir,
        {
            "name": "imob-ai-intent-baseline-pt",
            "model_sha256": model_hash,
            "dataset_content_sha256": dataset_hash,
            "training_records": 320,
            "training_splits": ["train", "validation"],
            "labels": LABELS,
            "classifier_classes": classifier_classes,
            "trusted_types": untrusted,
            "runtime": {
                "python": ".".join(map(str, sys.version_info[:3])),
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            },
            "frozen_metrics": FROZEN_METRICS,
        },
    )
    return manifest
