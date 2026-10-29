"""Configurações candidatas de TF-IDF definidas antes da avaliação."""

from dataclasses import dataclass
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    kind: str
    ngram_range: tuple[int, int]
    min_df: int
    c: float
    class_weight: Optional[str]
    complexity_rank: int


CANDIDATE_CONFIGS = (
    FeatureConfig("word_unigram", "word", (1, 1), 1, 1.0, None, 0),
    FeatureConfig("word_bigram", "word", (1, 2), 1, 1.0, None, 1),
    FeatureConfig("char_wb", "char", (3, 5), 2, 1.0, None, 2),
    FeatureConfig("word_char", "combined", (1, 2), 1, 1.0, None, 3),
)


def build_features(config: FeatureConfig) -> object:
    if config.kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )
    if config.kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
        ]
    )
