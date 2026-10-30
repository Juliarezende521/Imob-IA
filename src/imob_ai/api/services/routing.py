"""Roteamento fixo, transparente e não generativo."""

from dataclasses import dataclass

from imob_ai.api.schemas import RouteAction


@dataclass(frozen=True)
class RouteDecision:
    action: RouteAction
    message: str


ROUTES = {
    "AGENDAR_VISITA": RouteDecision(
        RouteAction.HUMAN_SUPPORT,
        "O agendamento deve ser continuado com atendimento humano.",
    ),
    "DUVIDA_IMOVEL": RouteDecision(
        RouteAction.IDENTIFY_PROPERTY,
        "Informe o identificador do imóvel para que a dúvida possa ser direcionada.",
    ),
    "DOCUMENTACAO_LOCACAO": RouteDecision(
        RouteAction.PROVIDE_GENERIC_DOCUMENT_GUIDANCE,
        "A documentação varia conforme a análise; solicite a lista oficial "
        "ao atendimento.",
    ),
    "FINANCEIRO_LOCACAO": RouteDecision(
        RouteAction.HUMAN_FINANCIAL_SUPPORT,
        "Questões financeiras devem ser tratadas por atendimento humano autorizado.",
    ),
    "MANUTENCAO_IMOVEL": RouteDecision(
        RouteAction.HUMAN_MAINTENANCE_SUPPORT,
        "A solicitação de manutenção será direcionada para atendimento humano.",
    ),
    "ATENDIMENTO_HUMANO": RouteDecision(
        RouteAction.HUMAN_SUPPORT,
        "A solicitação deve ser continuada com uma pessoa.",
    ),
    "FORA_DE_ESCOPO": RouteDecision(
        RouteAction.OUT_OF_SCOPE,
        "Este assistente atende somente solicitações imobiliárias previstas.",
    ),
}


def decide_route(intent: str, requires_review: bool) -> RouteDecision:
    if requires_review:
        return RouteDecision(
            RouteAction.HUMAN_REVIEW,
            "A classificação ficou incerta e precisa de revisão humana.",
        )
    if intent == "BUSCA_IMOVEL":
        return RouteDecision(
            RouteAction.SEARCH_PROPERTIES,
            "Busca lexical executada no catálogo sintético.",
        )
    return ROUTES[intent]
