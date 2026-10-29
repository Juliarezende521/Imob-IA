"""Configuração central da busca semântica."""

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_DIMENSION = 384
MODEL_LICENSE = "Apache-2.0"
SENTENCE_TRANSFORMERS_VERSION = "5.1.2"

# Selecionado exclusivamente nas 20 consultas de development.
SEMANTIC_THRESHOLD = 0.20
THRESHOLD_CANDIDATES = tuple(value / 100 for value in range(0, 81, 5))
