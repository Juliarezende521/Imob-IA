"""Renderização do relatório técnico de classificação."""

from typing import Any

from imob_ai.intent.schemas import Intent

LABELS = [intent.value for intent in Intent]


def render_evaluation_report(results: dict[str, Any]) -> str:
    selected = results["selected_config"]
    lines = [
        "# Avaliação do classificador de intenções",
        "",
        "O dataset é inteiramente sintético. O teste foi avaliado somente após "
        "o congelamento do pipeline em validação.",
        "",
        "## Configurações avaliadas em validação",
        "",
        "| Configuração | Accuracy | Precision macro | Recall macro | F1 macro | "
        "Menor F1 | Dimensões |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results["candidate_results"]:
        lines.append(
            f"| {item['name']} | {item['accuracy']:.3f} | "
            f"{item['precision_macro']:.3f} | {item['recall_macro']:.3f} | "
            f"{item['f1_macro']:.3f} | {item['min_class_f1']:.3f} | "
            f"{item['dimensions']} |"
        )
    lines.extend(
        [
            "",
            "## Pipeline congelado",
            "",
            f"- configuração: `{selected['name']}`;",
            f"- tipo: `{selected['kind']}`;",
            f"- n-gramas principais: `{tuple(selected['ngram_range'])}`;",
            f"- C: `{selected['c']}`;",
            "- classificador: Regressão Logística;",
            "- seed: `42`;",
            "- scikit-learn: `1.6.1`.",
            "",
            "## Métricas agregadas",
            "",
            "| Conjunto | Modelo | Accuracy | Precision macro | Recall macro | "
            "F1 macro | F1 weighted |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            _metric_row("Validação", "Dummy", results["validation_dummy"]),
            _metric_row("Validação", "TF-IDF + LogReg", results["validation"]),
            _metric_row("Teste", "Dummy", results["test_dummy"]),
            _metric_row("Teste", "TF-IDF + LogReg", results["test"]),
            "",
            "## Métricas por classe no teste",
            "",
            "| Classe | Precision | Recall | F1 | Suporte |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in LABELS:
        metrics = results["test"]["per_class"][label]
        lines.append(
            f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {int(metrics['support'])} |"
        )
    lines.extend(["", "## Matriz de confusão do teste", ""])
    class_order = ", ".join(f"`{label}`" for label in LABELS)
    lines.append(f"Ordem das classes: {class_order}.")
    lines.extend(["", "```text"])
    lines.extend(str(row) for row in results["test"]["confusion_matrix"])
    lines.extend(["```", "", "## Confusões mais frequentes", ""])
    if results["confusions"]:
        for item in results["confusions"]:
            lines.append(
                f"- {item['expected']} → {item['predicted']}: {item['count']} caso(s)."
            )
    else:
        lines.append("- Nenhuma confusão no teste atual.")
    lines.extend(["", "## Erros individuais", ""])
    for error in results["errors"]:
        scores = ", ".join(
            f"{label}={score:.3f}" for label, score in error["top_scores"]
        )
        lines.append(
            f"- `{error['id']}`: “{error['text']}” — esperado "
            f"`{error['expected']}`, previsto `{error['predicted']}`; scores: "
            f"{scores}; família `{error['scenario_family']}`; "
            f"dificuldade `{error['difficulty']}`. Possível causa: vocabulário "
            "compartilhado e cenário não observado no treino."
        )
    lines.extend(
        [
            "",
            "## Limitações",
            "",
            "- mensagens sintéticas não representam toda a linguagem real;",
            "- apenas uma família por classe existe em validação e teste;",
            "- scores de `predict_proba` não são certeza e não foram calibrados;",
            "- erros ortográficos e ambiguidades reais podem ser mais variados;",
            "- o classificador não está pronto para produção.",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_row(split: str, model: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {split} | {model} | {metrics['accuracy']:.3f} | "
        f"{metrics['precision_macro']:.3f} | {metrics['recall_macro']:.3f} | "
        f"{metrics['f1_macro']:.3f} | {metrics['f1_weighted']:.3f} |"
    )
