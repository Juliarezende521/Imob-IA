"""Classificação de intenção pelo baseline tradicional."""

from fastapi import APIRouter, Depends

from imob_ai.api.dependencies import AppServices, require_services
from imob_ai.api.schemas import ClassificationResponse, MessageRequest

router = APIRouter(prefix="/v1/intents", tags=["intents"])


def response_from_classification(classification) -> ClassificationResponse:
    return ClassificationResponse(
        intent=classification.intent,
        model_score=classification.model_score,
        second_intent=classification.second_intent,
        second_model_score=classification.second_model_score,
        margin=classification.margin,
        requires_human_review=classification.requires_human_review,
        review_reason=classification.review_reason,
    )


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classifica uma mensagem sintética ou local",
)
def classify(
    request: MessageRequest,
    services: AppServices = Depends(require_services),
) -> ClassificationResponse:
    result = services.classifier.classify(request.message)
    return response_from_classification(result)
