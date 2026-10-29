# Busca estruturada e baseline lexical

## O que este marco implementa

Esta busca é um baseline determinístico, não um modelo de Inteligência
Artificial. Ela oferece uma referência simples para comparar futuramente com
uma busca semântica baseada em embeddings.

Os filtros estruturados são combinados com lógica `AND`: todo filtro preenchido
precisa ser satisfeito. Imóveis indisponíveis são excluídos por padrão.

## Filtros

A consulta aceita finalidade, tipo, cidade, bairro, faixas de preço e área,
quantidades mínimas, aceitação de animais, mobília, características obrigatórias,
disponibilidade, texto livre, ordenação e limite.

Cidade, bairro, características e texto são comparados sem diferença de caixa,
acentos ou espaços excedentes. Assim, `Ribeirao Preto` encontra
`Ribeirão Preto`.

## Pontuação lexical

O texto pesquisável reúne finalidade, tipo, cidade, bairro, características e
descrição. A consulta e o texto são normalizados e separados em termos
alfanuméricos distintos. Cada termo da consulta encontrado vale um ponto.

Uma consulta com três termos pode, portanto, obter pontuação de zero a três. A
pontuação **não é probabilidade**, confiança ou medida de significado. Pelo
menos um termo precisa aparecer para que um imóvel participe do resultado.

Na ordenação por relevância, os critérios de desempate são menor preço e depois
identificador. As demais ordenações usam preço ou área e identificador como
desempate, mantendo o resultado determinístico.

## Exemplos

```python
from imob_ai.data import load_catalog
from imob_ai.search import PropertySearchQuery, search_properties

catalog = load_catalog("data/synthetic/properties.json")

# Casas para aluguel em Franca
search_properties(
    catalog,
    PropertySearchQuery(purpose="aluguel", property_type="casa", city="Franca"),
)

# Imóveis de até R$ 2.500
search_properties(catalog, PropertySearchQuery(max_price=2500))

# Imóveis que aceitam animais
search_properties(catalog, PropertySearchQuery(pets_allowed=True))

# Busca lexical
search_properties(catalog, PropertySearchQuery(text="área gourmet"))

# Resultado vazio
search_properties(catalog, PropertySearchQuery(city="Cidade Inexistente"))
```

## Limitações

- sinônimos não são compreendidos: `garagem` e `estacionamento` são termos
  diferentes;
- erros de digitação não são corrigidos;
- a pontuação conta termos, sem ponderar raridade ou importância;
- palavras isoladas não representam contexto ou intenção;
- ainda não existe conjunto de relevância nem métrica de qualidade calculada.

A futura busca semântica somente será considerada melhor depois de uma avaliação
reproduzível contra este baseline.
