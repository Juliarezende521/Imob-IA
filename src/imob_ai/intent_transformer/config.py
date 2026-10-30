"""Constantes previamente definidas para o experimento Transformer."""

from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_ID = "distilbert/distilbert-base-multilingual-cased"
MODEL_REVISION = "45c032ab32cc946ad88a166f7cb282f58c753c2e"
MODEL_LICENSE = "Apache-2.0"
RANDOM_SEED = 42
MAX_LENGTH = 64
MAX_EPOCHS = 5
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 2
SIMPLICITY_TIE_EPSILON = 0.005

ARTIFACT_ROOT = Path("artifacts/intent-transformer")
FINAL_MODEL_PATH = ARTIFACT_ROOT / "final-model"
SELECTION_PATH = ARTIFACT_ROOT / "selection.json"
TRAINING_METADATA_PATH = ARTIFACT_ROOT / "training-metadata.json"
FROZEN_CONFIG_PATH = Path("docs/experiments/intent-transformer-frozen-config.json")
FROZEN_CONFIG_HASH_PATH = Path(
    "docs/experiments/intent-transformer-frozen-config.sha256"
)
TEST_RESULTS_PATH = Path("docs/experiments/intent-transformer-test-results.json")
BENCHMARK_PATH = Path("docs/experiments/intent-transformer-benchmark.json")
CHALLENGE_RECEIPT_PATH = Path(
    "docs/experiments/intent-challenge-evaluation-receipt.json"
)


@dataclass(frozen=True)
class Strategy:
    name: str
    freeze_encoder: bool
    learning_rate: float


STRATEGIES = (
    Strategy("frozen_encoder", True, 5e-4),
    Strategy("full_fine_tuning", False, 2e-5),
)


def strategy_dict(strategy: Strategy) -> dict[str, object]:
    return asdict(strategy)
