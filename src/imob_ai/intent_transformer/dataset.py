"""Dataset PyTorch sem descoberta genérica de arquivos."""

from collections.abc import Sequence
from typing import Any

from imob_ai.intent_transformer.labels import LABEL2ID


class IntentTorchDataset:
    def __init__(self, records: Sequence[Any], tokenizer: Any, max_length: int):
        split_values = {
            getattr(getattr(item, "split", None), "value", getattr(item, "split", None))
            for item in records
        }
        if "challenge" in split_values:
            raise ValueError("Challenge não pode entrar em dataset de treinamento.")
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = self.records[index]
        encoded = self.tokenizer(
            item.text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        result = {key: value.squeeze(0) for key, value in encoded.items()}
        result["labels"] = torch.tensor(LABEL2ID[item.intent.value], dtype=torch.long)
        return result
