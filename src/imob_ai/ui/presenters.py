"""Transformações puras para apresentação na interface."""

from imob_ai.ui.schemas import Classification, Property

SCORE_EXPLANATION = (
    "O model_score representa a força relativa da decisão do classificador e "
    "não uma probabilidade calibrada."
)

ACTION_LABELS = {
    "search_properties": "Buscar imóveis",
    "human_review": "Revisão humana",
    "identify_property": "Identificar o imóvel",
    "provide_generic_document_guidance": "Orientação documental geral",
    "human_financial_support": "Atendimento financeiro humano",
    "human_maintenance_support": "Atendimento de manutenção humano",
    "human_support": "Atendimento humano",
    "out_of_scope": "Fora do escopo",
}


def format_currency(value: float) -> str:
    grouped = f"{value:,.2f}"
    return "R$ " + grouped.replace(",", "X").replace(".", ",").replace("X", ".")


def format_area(value: float) -> str:
    rendered = f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{rendered} m²"


def format_score(value: float) -> str:
    return f"{value:.3f}"


def uncertainty_label(classification: Classification) -> str:
    if classification.requires_human_review:
        return "Resultado incerto — encaminhamento humano recomendado"
    return "Resultado aceito pela política operacional"


def human_follow_up(action: str) -> bool:
    return action in {
        "human_review",
        "human_financial_support",
        "human_maintenance_support",
        "human_support",
    }


def property_summary(property_: Property) -> str:
    return (
        f"{property_.property_type.value.title()} para "
        f"{property_.purpose.value} em {property_.city}"
    )
