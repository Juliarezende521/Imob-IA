"""Execução e apresentação da avaliação do baseline lexical."""

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean
from typing import Optional

from imob_ai.data import Property, load_catalog
from imob_ai.evaluation.dataset import load_evaluation_cases
from imob_ai.evaluation.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from imob_ai.evaluation.schemas import (
    AggregateMetrics,
    EvaluationCase,
    EvaluationReport,
    QueryEvaluation,
)
from imob_ai.search import search_properties

DEFAULT_CATALOG_PATH = Path("data/synthetic/properties.json")
DEFAULT_QUERIES_PATH = Path("data/evaluation/retrieval_queries.json")


def evaluate_cases(
    catalog: Sequence[Property],
    cases: Sequence[EvaluationCase],
    retriever: Optional[
        Callable[[Sequence[Property], EvaluationCase], list[str]]
    ] = None,
) -> EvaluationReport:
    """Executa o baseline e agrega métricas sem alterar busca ou julgamentos."""
    query_results = [_evaluate_case(catalog, case, retriever) for case in cases]

    split_groups: dict[str, list[QueryEvaluation]] = defaultdict(list)
    category_groups: dict[str, list[QueryEvaluation]] = defaultdict(list)
    for result in query_results:
        split_groups[result.split.value].append(result)
        category_groups[result.category.value].append(result)

    return EvaluationReport(
        queries=query_results,
        overall=_aggregate(query_results),
        by_split={
            name: _aggregate(group) for name, group in sorted(split_groups.items())
        },
        by_category={
            name: _aggregate(group) for name, group in sorted(category_groups.items())
        },
    )


def render_markdown(
    report: EvaluationReport, title: str = "Avaliação do baseline lexical"
) -> str:
    """Produz relatório Markdown determinístico."""
    lines = [
        f"# {title}",
        "",
        "Este relatório mede o baseline determinístico sobre julgamentos manuais.",
        "Não há embeddings ou modelo de Inteligência Artificial nesta etapa.",
        "",
        "## Resumo",
        "",
        _metrics_table_row("Geral", report.overall, header=True),
        _metrics_table_row("Geral", report.overall),
    ]
    for name, metrics in report.by_split.items():
        lines.append(_metrics_table_row(name, metrics))

    lines.extend(["", "## Resultados por categoria", ""])
    lines.append(_metrics_table_row("Categoria", report.overall, header=True))
    for name, metrics in report.by_category.items():
        lines.append(_metrics_table_row(name, metrics))

    retrieval = [
        result for result in report.queries if result.reciprocal_rank is not None
    ]
    ranked = sorted(
        retrieval,
        key=lambda result: (
            -(result.reciprocal_rank or 0),
            -(result.recall_at_5 or 0),
            result.case_id,
        ),
    )
    lines.extend(["", "## Melhores resultados", ""])
    lines.extend(_query_line(result) for result in ranked[:5])
    lines.extend(["", "## Piores resultados", ""])
    lines.extend(_query_line(result) for result in reversed(ranked[-5:]))

    failures = [
        result
        for result in report.queries
        if result.reciprocal_rank == 0 or result.no_result_correct is False
    ]
    lines.extend(["", "## Falhas individuais", ""])
    lines.extend(_query_line(result) for result in failures)
    if not failures:
        lines.append("- Nenhuma falha segundo os julgamentos atuais.")

    lines.extend(
        [
            "",
            "## Fórmulas e denominadores",
            "",
            "- `Precision@K = relevantes recuperados nas K primeiras posições / K`.",
            "- `Recall@K = relevantes recuperados nas K primeiras posições "
            "/ total de relevantes`.",
            "- `MRR = média de 1 / posição do primeiro relevante`.",
            "- `Hit Rate@K = consultas com ao menos um relevante no top K "
            "/ consultas de recuperação`.",
            "- `No-result accuracy = consultas corretamente vazias "
            "/ consultas sem resultado esperado`.",
            "- Consultas sem resultado esperado não entram nas médias de "
            "Precision, Recall, MRR ou Hit Rate.",
            "",
            "## Regra contra vazamento",
            "",
            "O split de desenvolvimento pode orientar ajustes futuros. O split de "
            "teste deve permanecer estável e não deve ser consultado repetidamente "
            "para ajustar implementações. Mudanças posteriores nos julgamentos "
            "precisam de justificativa e versionamento.",
            "",
            "## Limitações",
            "",
            "- catálogo pequeno e totalmente sintético;",
            "- julgamentos manuais e parcialmente subjetivos;",
            "- poucas consultas por categoria;",
            "- ausência de avaliadores independentes;",
            "- métricas válidas para comparação interna, não para generalização "
            "ao mercado real.",
            "",
        ]
    )
    return "\n".join(lines)


def _evaluate_case(
    catalog: Sequence[Property],
    case: EvaluationCase,
    retriever: Optional[Callable[[Sequence[Property], EvaluationCase], list[str]]],
) -> QueryEvaluation:
    if retriever is None:
        query = case.filters.model_copy(update={"text": case.query, "limit": None})
        search_result = search_properties(catalog, query)
        retrieved_ids = [item.property.id for item in search_result.items]
    else:
        retrieved_ids = retriever(catalog, case)

    if case.expected_no_results:
        return QueryEvaluation(
            case_id=case.id,
            split=case.split,
            category=case.category,
            relevant_ids=[],
            retrieved_ids=retrieved_ids,
            no_result_correct=not retrieved_ids,
        )

    relevant = set(case.relevant_property_ids)
    return QueryEvaluation(
        case_id=case.id,
        split=case.split,
        category=case.category,
        relevant_ids=case.relevant_property_ids,
        retrieved_ids=retrieved_ids,
        precision_at_1=precision_at_k(retrieved_ids, relevant, 1),
        precision_at_3=precision_at_k(retrieved_ids, relevant, 3),
        precision_at_5=precision_at_k(retrieved_ids, relevant, 5),
        recall_at_1=recall_at_k(retrieved_ids, relevant, 1),
        recall_at_3=recall_at_k(retrieved_ids, relevant, 3),
        recall_at_5=recall_at_k(retrieved_ids, relevant, 5),
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant),
        hit_at_1=hit_at_k(retrieved_ids, relevant, 1),
        hit_at_3=hit_at_k(retrieved_ids, relevant, 3),
        hit_at_5=hit_at_k(retrieved_ids, relevant, 5),
    )


def _aggregate(results: Sequence[QueryEvaluation]) -> AggregateMetrics:
    retrieval = [result for result in results if result.reciprocal_rank is not None]
    no_result = [result for result in results if result.no_result_correct is not None]
    return AggregateMetrics(
        total_cases=len(results),
        retrieval_cases=len(retrieval),
        no_result_cases=len(no_result),
        precision_at_1=_mean(retrieval, "precision_at_1"),
        precision_at_3=_mean(retrieval, "precision_at_3"),
        precision_at_5=_mean(retrieval, "precision_at_5"),
        recall_at_1=_mean(retrieval, "recall_at_1"),
        recall_at_3=_mean(retrieval, "recall_at_3"),
        recall_at_5=_mean(retrieval, "recall_at_5"),
        mrr=_mean(retrieval, "reciprocal_rank"),
        hit_rate_at_1=_mean(retrieval, "hit_at_1"),
        hit_rate_at_3=_mean(retrieval, "hit_at_3"),
        hit_rate_at_5=_mean(retrieval, "hit_at_5"),
        no_result_accuracy=(
            fmean(float(result.no_result_correct) for result in no_result)
            if no_result
            else None
        ),
    )


def _mean(results: Sequence[QueryEvaluation], field: str) -> Optional[float]:
    values = [getattr(result, field) for result in results]
    return fmean(values) if values else None


def _format(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def _metrics_table_row(
    name: str, metrics: AggregateMetrics, header: bool = False
) -> str:
    if header:
        columns = (
            "| Grupo | Casos | Recuperação | Ausência | P@1 | P@3 | P@5 | R@1 | "
            "R@3 | R@5 | MRR | Hit@5 | Acc. ausência |"
        )
        separator = (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        )
        return f"{columns}\n{separator}"
    return (
        f"| {name} | {metrics.total_cases} | {metrics.retrieval_cases} | "
        f"{metrics.no_result_cases} | {_format(metrics.precision_at_1)} | "
        f"{_format(metrics.precision_at_3)} | {_format(metrics.precision_at_5)} | "
        f"{_format(metrics.recall_at_1)} | {_format(metrics.recall_at_3)} | "
        f"{_format(metrics.recall_at_5)} | {_format(metrics.mrr)} | "
        f"{_format(metrics.hit_rate_at_5)} | {_format(metrics.no_result_accuracy)} |"
    )


def _query_line(result: QueryEvaluation) -> str:
    if result.no_result_correct is not None:
        return (
            f"- `{result.case_id}`: ausência correta = "
            f"`{result.no_result_correct}`; recuperados = {result.retrieved_ids}."
        )
    return (
        f"- `{result.case_id}`: MRR = {_format(result.reciprocal_rank)}, "
        f"Recall@5 = {_format(result.recall_at_5)}; "
        f"recuperados = {result.retrieved_ids[:5]}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--engine", choices=["lexical", "semantic", "compare"], default="lexical"
    )
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    cases = load_evaluation_cases(args.queries, catalog)
    lexical_report = evaluate_cases(catalog, cases)
    if args.engine == "lexical":
        markdown = render_markdown(lexical_report)
    else:
        from imob_ai.semantic.config import SEMANTIC_THRESHOLD
        from imob_ai.semantic.encoder import SentenceTransformerEncoder
        from imob_ai.semantic.report import render_comparison
        from imob_ai.semantic.service import SemanticSearchService

        service = SemanticSearchService(SentenceTransformerEncoder())

        def semantic_retriever(
            items: Sequence[Property], case: EvaluationCase
        ) -> list[str]:
            query = case.filters.model_copy(update={"text": case.query, "limit": None})
            result = service.search(items, query, threshold=SEMANTIC_THRESHOLD)
            return [match.property.id for match in result.items]

        semantic_report = evaluate_cases(catalog, cases, semantic_retriever)
        if args.engine == "semantic":
            markdown = render_markdown(
                semantic_report, title="Avaliação da busca semântica"
            )
        else:
            markdown = render_comparison(lexical_report, semantic_report)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
