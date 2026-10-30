"""Abertura única e auditável do challenge congelado."""

import argparse
import json

from imob_ai.intent.challenge import (
    CHALLENGE_PATH,
    load_challenge,
    sha256_file,
    verify_manifest,
)
from imob_ai.intent_transformer.config import CHALLENGE_RECEIPT_PATH
from imob_ai.intent_transformer.runtime import (
    baseline_predictions,
    evaluation_payload,
    load_and_verify_frozen,
    load_local_transformer,
    transformer_predictions,
)
from imob_ai.intent_transformer.train import choose_device


def guard_challenge_access(
    confirmed: bool,
    receipt_exists: bool,
    challenge_hash: str,
    frozen_challenge_hash: str,
) -> None:
    if not confirmed:
        raise RuntimeError("Confirmação explícita da configuração é obrigatória.")
    if receipt_exists:
        raise RuntimeError("O challenge já foi avaliado; receipt existente.")
    if challenge_hash != frozen_challenge_hash:
        raise RuntimeError("Hash do challenge diverge da configuração congelada.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-frozen-config", action="store_true")
    args = parser.parse_args()
    config, metadata = load_and_verify_frozen()
    records = load_challenge()
    verify_manifest(records)
    challenge_hash = sha256_file(CHALLENGE_PATH)
    guard_challenge_access(
        args.confirm_frozen_config,
        CHALLENGE_RECEIPT_PATH.exists(),
        challenge_hash,
        config["dataset_hashes"]["challenge"],
    )

    baseline_expected, baseline_predicted = baseline_predictions(records)
    device = choose_device()
    tokenizer, model = load_local_transformer(device)
    expected, predicted, _ = transformer_predictions(
        records, tokenizer, model, device, config["max_length"]
    )
    baseline = evaluation_payload(records, baseline_expected, baseline_predicted)
    transformer = evaluation_payload(records, expected, predicted)
    receipt = {
        "schema_version": "1",
        "first_evaluation": True,
        "challenge_sha256": challenge_hash,
        "frozen_config_sha256": metadata["frozen_config_sha256"],
        "artifact_sha256": metadata["artifact_sha256"],
        "versions": config["versions"],
        "device": device,
        "baseline": baseline,
        "transformer": transformer,
        "f1_macro_difference": (
            transformer["metrics"]["f1_macro"] - baseline["metrics"]["f1_macro"]
        ),
        "declaration": (
            "Primeira e única avaliação do challenge; nenhuma configuração, "
            "hiperparâmetro, tokenizer, modelo ou dado será alterado após o resultado."
        ),
    }
    CHALLENGE_RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHALLENGE_RECEIPT_PATH.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
