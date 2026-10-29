"""Relatório comparativo lexical versus semântico."""

from typing import Optional

from imob_ai.evaluation.schemas import (
    AggregateMetrics,
    EvaluationReport,
    QueryEvaluation,
)
from imob_ai.semantic.config import (
    MODEL_DIMENSION,
    MODEL_LICENSE,
    MODEL_NAME,
    SEMANTIC_THRESHOLD,
    SENTENCE_TRANSFORMERS_VERSION,
)

METRICS = [
    ("Precision@1", "precision_at_1"),
    ("Precision@3", "precision_at_3"),
    ("Precision@5", "precision_at_5"),
    ("Recall@1", "recall_at_1"),
    ("Recall@3", "recall_at_3"),
    ("Recall@5", "recall_at_5"),
    ("MRR", "mrr"),
    ("Hit Rate@5", "hit_rate_at_5"),
    ("Acerto de ausência", "no_result_accuracy"),
]


def render_comparison(lexical: EvaluationReport, semantic: EvaluationReport) -> str:
    """Compara relatórios calculados sobre os mesmos casos."""
    lines = [
        "# Avaliação da busca semântica",
        "",
        "## Configuração",
        "",
        f"- modelo: `{MODEL_NAME}`;",
        f"- dimensão: {MODEL_DIMENSION};",
        f"- licença do modelo: {MODEL_LICENSE};",
        f"- sentence-transformers: {SENTENCE_TRANSFORMERS_VERSION};",
        "- similaridade: cosseno exato em memória;",
        f"- limiar calibrado em development: {SEMANTIC_THRESHOLD:.3f}.",
        "",
        "## Comparação geral",
        "",
        *_comparison_table(lexical.overall, semantic.overall),
    ]
    for split in ("development", "test"):
        lines.extend(
            [
                "",
                f"## Split {split}",
                "",
                *_comparison_table(lexical.by_split[split], semantic.by_split[split]),
            ]
        )

    lines.extend(["", "## Resultados por categoria", ""])
    lines.append("| Categoria | MRR lexical | MRR semântico | Diferença |")
    lines.append("| --- | ---: | ---: | ---: |")
    for category in sorted(lexical.by_category):
        left = lexical.by_category[category].mrr
        right = semantic.by_category[category].mrr
        lines.append(
            f"| {category} | {_fmt(left)} | {_fmt(right)} | {_delta(left, right)} |"
        )

    improved, regressed, tied = _case_differences(lexical, semantic)
    lines.extend(
        [
            "",
            "## Diferenças por consulta",
            "",
            f"- melhorias: {improved or ['nenhuma']};",
            f"- regressões: {regressed or ['nenhuma']};",
            f"- empates: {len(tied)} consultas.",
            "",
            "## Interpretação e limitações",
            "",
            "A comparação usa apenas 32 consultas e 48 imóveis sintéticos. O split "
            "test é pequeno; diferenças são preliminares. Similaridade não é "
            "probabilidade. O limiar foi escolhido somente em development e pode "
            "não generalizar. Não há busca híbrida, índice vetorial ou RAG.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_table(
    lexical: AggregateMetrics, semantic: AggregateMetrics
) -> list[str]:
    lines = [
        "| Métrica | Lexical | Semântico | Diferença |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, field in METRICS:
        left = getattr(lexical, field)
        right = getattr(semantic, field)
        lines.append(
            f"| {label} | {_fmt(left)} | {_fmt(right)} | {_delta(left, right)} |"
        )
    return lines


def _case_differences(
    lexical: EvaluationReport, semantic: EvaluationReport
) -> tuple[list[str], list[str], list[str]]:
    improved: list[str] = []
    regressed: list[str] = []
    tied: list[str] = []
    semantic_by_id = {result.case_id: result for result in semantic.queries}
    for left in lexical.queries:
        right = semantic_by_id[left.case_id]
        left_score = _case_score(left)
        right_score = _case_score(right)
        if right_score > left_score:
            improved.append(left.case_id)
        elif right_score < left_score:
            regressed.append(left.case_id)
        else:
            tied.append(left.case_id)
    return improved, regressed, tied


def _case_score(result: QueryEvaluation) -> float:
    if result.no_result_correct is not None:
        return float(result.no_result_correct)
    reciprocal_rank = result.reciprocal_rank or 0.0
    recall_at_5 = result.recall_at_5 or 0.0
    return (reciprocal_rank + recall_at_5) / 2


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def _delta(left: Optional[float], right: Optional[float]) -> str:
    if left is None or right is None:
        return "—"
    return f"{right - left:+.3f}"
