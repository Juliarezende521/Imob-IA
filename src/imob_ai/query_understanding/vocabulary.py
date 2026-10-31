"""Vocabulário controlado da compreensão determinística."""

from imob_ai.data.schemas import PropertyPurpose, PropertyType

PURPOSE_ALIASES = {
    "aluguel": PropertyPurpose.RENT,
    "alugar": PropertyPurpose.RENT,
    "alugando": PropertyPurpose.RENT,
    "locacao": PropertyPurpose.RENT,
    "locar": PropertyPurpose.RENT,
    "comprar": PropertyPurpose.SALE,
    "compra": PropertyPurpose.SALE,
    "venda": PropertyPurpose.SALE,
    "a venda": PropertyPurpose.SALE,
}

PROPERTY_TYPE_ALIASES = {
    "casa": PropertyType.HOUSE,
    "casas": PropertyType.HOUSE,
    "acasa": PropertyType.HOUSE,
    "apartamento": PropertyType.APARTMENT,
    "apartamentos": PropertyType.APARTMENT,
    "ape": PropertyType.APARTMENT,
    "terreno": PropertyType.LAND,
    "terrenos": PropertyType.LAND,
    "comercial": PropertyType.COMMERCIAL,
    "sala comercial": PropertyType.COMMERCIAL,
    "ponto comercial": PropertyType.COMMERCIAL,
    "imovel comercial": PropertyType.COMMERCIAL,
}

CONSERVATIVE_CORRECTIONS = {"acasa": "casa"}

STOPWORDS = {
    "a",
    "ao",
    "as",
    "com",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "eu",
    "imovel",
    "me",
    "na",
    "no",
    "o",
    "os",
    "para",
    "por",
    "preciso",
    "procuro",
    "que",
    "quero",
    "um",
    "uma",
}

NON_LOCATION_TERMS = {
    "area",
    "condominio",
    "garagem",
    "mobiliado",
    "mobiliada",
    "quintal",
}
