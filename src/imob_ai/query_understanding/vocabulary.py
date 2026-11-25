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

MONEY_MULTIPLIERS = {
    "mil": 1_000,
    "milhao": 1_000_000,
    "milhoes": 1_000_000,
}

MIN_QUANTITY_NOUNS = {
    "quarto": "min_bedrooms",
    "quartos": "min_bedrooms",
    "dormitorio": "min_bedrooms",
    "dormitorios": "min_bedrooms",
    "banheiro": "min_bathrooms",
    "banheiros": "min_bathrooms",
}

MIN_QUANTITY_PREFIXES = (
    ("a", "partir", "de"),
    ("pelo", "menos"),
    ("no", "minimo"),
    ("com",),
    ("e",),
)

# Limites máximos sem filtro equivalente na busca: viram esclarecimento.
UPPER_BOUND_NOUNS = {
    "quarto": "bedrooms",
    "quartos": "bedrooms",
    "dormitorio": "bedrooms",
    "dormitorios": "bedrooms",
    "banheiro": "bathrooms",
    "banheiros": "bathrooms",
    "vaga": "parking_spaces",
    "vagas": "parking_spaces",
    "m": "area_m2",
    "m2": "area_m2",
    "metros": "area_m2",
}

# Expressões após "em" que descrevem o imóvel, não uma localização.
NON_LOCATION_PREFIXES = (
    ("bom", "estado"),
    ("otimo", "estado"),
    ("excelente", "estado"),
    ("perfeito", "estado"),
    ("boas", "condicoes"),
    ("otimas", "condicoes"),
    ("perfeitas", "condicoes"),
)

NON_LOCATION_TERMS = {
    "area",
    "condominio",
    "garagem",
    "mobiliado",
    "mobiliada",
    "quintal",
}
