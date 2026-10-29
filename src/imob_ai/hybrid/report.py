"""Relatório comparativo lexical, semântico e híbrido."""

from typing import Optional

from imob_ai.evaluation.schemas import (
    AggregateMetrics,
    EvaluationReport,
    QueryEvaluation,
)
from imob_ai.hybrid.config import (
    CALIBRATION_RESULTS,
    HYBRID_LEXICAL_WEIGHT,
    RRF_K,
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


def render_hybrid_comparison(
    lexical: EvaluationReport,
    semantic: EvaluationReport,
    hybrid: EvaluationReport,
) -> str:
    """Renderiza comparação após congelamento da configuração híbrida."""
    lines = [
        "# Avaliação da busca híbrida",
        "",
        "## Configuração congelada antes do teste",
        "",
        "- estratégia: Weighted Reciprocal Rank Fusion;",
        f"- peso lexical: {HYBRID_LEXICAL_WEIGHT:.1f};",
        f"- peso semântico: {1 - HYBRID_LEXICAL_WEIGHT:.1f};",
        f"- constante RRF k: {RRF_K};",
        "- candidatos: união completa depois dos filtros;",
        "- ausência em um ranking: posição ausente e contribuição zero;",
        "- evidência: correspondência lexical ou cosseno semântico >= 0,20;",
        "- desempate: score híbrido, semântico, lexical e ID.",
        "",
        "## Calibração em development",
        "",
        "| Peso lexical | Peso semântico | MRR | Recall@5 | Precision@5 | "
        "Hit@5 | Ausência |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for weight, mrr, recall, precision, hit, no_result in CALIBRATION_RESULTS:
        lines.append(
            f"| {weight:.1f} | {1 - weight:.1f} | {mrr:.3f} | {recall:.3f} | "
            f"{precision:.3f} | {hit:.3f} | {no_result:.3f} |"
        )

    lines.extend(["", "## Comparação geral", ""])
    lines.extend(_table(lexical.overall, semantic.overall, hybrid.overall))
    for split in ("development", "test"):
        lines.extend(["", f"## Split {split}", ""])
        lines.extend(
            _table(
                lexical.by_split[split],
                semantic.by_split[split],
                hybrid.by_split[split],
            )
        )

    lines.extend(["", "## Resultados por categoria", ""])
    lines.append(
        "| Categoria | MRR lexical | MRR semântico | MRR híbrido | Híbrido − lexical |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for category in sorted(lexical.by_category):
        left = lexical.by_category[category].mrr
        middle = semantic.by_category[category].mrr
        right = hybrid.by_category[category].mrr
        lines.append(
            f"| {category} | {_fmt(left)} | {_fmt(middle)} | {_fmt(right)} | "
            f"{_delta(left, right)} |"
        )

    improved, regressed, tied = _differences(lexical, hybrid)
    test_differences = [
        (
            label,
            _numeric_delta(
                getattr(lexical.by_split["test"], field),
                getattr(hybrid.by_split["test"], field),
            ),
        )
        for label, field in METRICS
    ]
    improved_test = [label for label, delta in test_differences if delta > 0]
    regressed_test = [label for label, delta in test_differences if delta < 0]
    if not improved_test and not regressed_test:
        test_conclusion = (
            "No split test, o híbrido empatou com o lexical em todas as métricas; "
            "portanto, não demonstrou melhora fora do desenvolvimento."
        )
    else:
        test_conclusion = (
            f"No split test, melhorou {improved_test or ['nenhuma métrica']} e "
            f"regrediu {regressed_test or ['nenhuma métrica']}."
        )
    lines.extend(
        [
            "",
            "## Diferenças do híbrido em relação ao lexical",
            "",
            f"- melhorias: {improved or ['nenhuma']};",
            f"- regressões: {regressed or ['nenhuma']};",
            f"- empates: {len(tied)} consultas.",
            "",
            "## Exemplos de contribuições diferentes",
            "",
            "- `QUERY-004`: `IMOB-027` não apareceu no ranking lexical, mas entrou "
            "pela posição semântica 2, com contribuição lexical zero;",
            "- `QUERY-013`: `IMOB-029` tinha posições lexical 4 e semântica 5; a "
            "combinação o levou à posição híbrida 3;",
            "- `QUERY-014`: `IMOB-048` não apareceu lexicalmente, mas entrou pela "
            "posição semântica 1, também com contribuição lexical zero.",
            "",
            "## Conclusão e limitações",
            "",
            test_conclusion,
            "",
            "A conclusão principal deve considerar o split test. O catálogo e o "
            "teste são pequenos e sintéticos; diferenças modestas não demonstram "
            "generalização. O score RRF não é probabilidade. A política de ausência "
            "possui alta incerteza por haver somente três casos em development.",
            "",
        ]
    )
    return "\n".join(lines)


def _table(
    lexical: AggregateMetrics,
    semantic: AggregateMetrics,
    hybrid: AggregateMetrics,
) -> list[str]:
    lines = [
        "| Métrica | Lexical | Semântico | Híbrido | Híbrido − Lexical |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, field in METRICS:
        left = getattr(lexical, field)
        middle = getattr(semantic, field)
        right = getattr(hybrid, field)
        lines.append(
            f"| {label} | {_fmt(left)} | {_fmt(middle)} | {_fmt(right)} | "
            f"{_delta(left, right)} |"
        )
    return lines


def _differences(
    lexical: EvaluationReport, hybrid: EvaluationReport
) -> tuple[list[str], list[str], list[str]]:
    improved: list[str] = []
    regressed: list[str] = []
    tied: list[str] = []
    hybrid_by_id = {result.case_id: result for result in hybrid.queries}
    for left in lexical.queries:
        right = hybrid_by_id[left.case_id]
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
    return ((result.reciprocal_rank or 0) + (result.recall_at_5 or 0)) / 2


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def _delta(left: Optional[float], right: Optional[float]) -> str:
    if left is None or right is None:
        return "—"
    return f"{right - left:+.3f}"


def _numeric_delta(left: Optional[float], right: Optional[float]) -> float:
    if left is None or right is None:
        return 0.0
    return right - left
