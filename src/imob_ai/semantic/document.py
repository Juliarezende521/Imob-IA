"""Representação textual canônica dos imóveis."""

from imob_ai.data import Property


def build_property_document(property_: Property) -> str:
    """Cria texto determinístico sem ID ou disponibilidade operacional."""
    features = ", ".join(property_.features) if property_.features else "nenhuma"
    return " | ".join(
        [
            f"finalidade: {property_.purpose.value}",
            f"tipo: {property_.property_type.value}",
            f"cidade: {property_.city}",
            f"bairro: {property_.neighborhood}",
            f"valor em reais: {property_.price:g}",
            f"área em metros quadrados: {property_.area_m2:g}",
            f"dormitórios: {property_.bedrooms}",
            f"banheiros: {property_.bathrooms}",
            f"vagas de garagem: {property_.parking_spaces}",
            f"aceita animais: {_yes_no(property_.pets_allowed)}",
            f"mobiliado: {_yes_no(property_.furnished)}",
            f"características: {features}",
            f"descrição: {property_.description}",
        ]
    )


def _yes_no(value: bool) -> str:
    return "sim" if value else "não"
