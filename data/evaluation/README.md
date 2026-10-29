# Conjunto de avaliação da recuperação

O arquivo `retrieval_queries.json` contém julgamentos manuais fixos para o
catálogo sintético. Cada caso registra consulta, filtros, imóveis relevantes,
split, categoria e justificativa.

O conjunto foi criado pela inspeção dos 48 registros. Somente imóveis
disponíveis são considerados, exceto quando `include_unavailable` for
explicitamente verdadeiro. Os julgamentos não foram derivados da posição
produzida pelo baseline.

## Desenvolvimento e teste

- `development`: pode orientar análises e ajustes futuros;
- `test`: deve permanecer estável e não deve ser usado repetidamente para
  adaptar a implementação.

Alterações futuras em consultas ou relevância precisam de justificativa e
versionamento. Isso reduz vazamento entre desenvolvimento e teste.

## Campos

- `id`: identificador único da consulta;
- `query`: texto enviado à busca lexical;
- `filters`: filtros estruturados aplicados junto com o texto;
- `relevant_property_ids`: todos os imóveis julgados razoavelmente relevantes;
- `split`: `development` ou `test`;
- `category`: tipo controlado de desafio;
- `notes`: justificativa do julgamento;
- `expected_no_results`: indica ausência de imóvel relevante.

O loader rejeita arquivo ausente, JSON inválido, lista vazia, IDs de consulta
repetidos, IDs relevantes repetidos ou inexistentes, valores controlados
inválidos, consulta vazia e contradições em casos sem resultado.

## Limitações

Os julgamentos são manuais, produzidos por uma única revisão e limitados a um
catálogo pequeno e fictício. Eles servem para comparação interna reproduzível,
não como evidência sobre o mercado imobiliário real.
