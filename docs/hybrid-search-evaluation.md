# Avaliação da busca híbrida

## Configuração congelada antes do teste

- estratégia: Weighted Reciprocal Rank Fusion;
- peso lexical: 0.9;
- peso semântico: 0.1;
- constante RRF k: 60;
- candidatos: união completa depois dos filtros;
- ausência em um ranking: posição ausente e contribuição zero;
- evidência: correspondência lexical ou cosseno semântico >= 0,20;
- desempate: score híbrido, semântico, lexical e ID.

## Calibração em development

| Peso lexical | Peso semântico | MRR | Recall@5 | Precision@5 | Hit@5 | Ausência |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1.0 | 0.850 | 0.705 | 0.635 | 0.882 | 0.667 |
| 0.1 | 0.9 | 0.851 | 0.705 | 0.635 | 0.882 | 0.667 |
| 0.2 | 0.8 | 0.856 | 0.705 | 0.635 | 0.882 | 0.667 |
| 0.3 | 0.7 | 0.871 | 0.764 | 0.647 | 0.941 | 0.667 |
| 0.4 | 0.6 | 0.897 | 0.793 | 0.659 | 1.000 | 0.667 |
| 0.5 | 0.5 | 0.897 | 0.793 | 0.659 | 1.000 | 0.667 |
| 0.6 | 0.4 | 0.902 | 0.793 | 0.659 | 1.000 | 0.667 |
| 0.7 | 0.3 | 0.902 | 0.764 | 0.647 | 1.000 | 0.667 |
| 0.8 | 0.2 | 0.902 | 0.764 | 0.647 | 1.000 | 0.667 |
| 0.9 | 0.1 | 0.922 | 0.764 | 0.647 | 1.000 | 0.667 |
| 1.0 | 0.0 | 0.917 | 0.764 | 0.647 | 1.000 | 0.667 |

## Comparação geral

| Métrica | Lexical | Semântico | Híbrido | Híbrido − Lexical |
| --- | ---: | ---: | ---: | ---: |
| Precision@1 | 0.923 | 0.885 | 0.923 | +0.000 |
| Precision@3 | 0.756 | 0.744 | 0.769 | +0.013 |
| Precision@5 | 0.623 | 0.631 | 0.646 | +0.023 |
| Recall@1 | 0.318 | 0.280 | 0.318 | +0.000 |
| Recall@3 | 0.634 | 0.596 | 0.654 | +0.019 |
| Recall@5 | 0.796 | 0.758 | 0.809 | +0.014 |
| MRR | 0.946 | 0.902 | 0.949 | +0.003 |
| Hit Rate@5 | 1.000 | 0.923 | 1.000 | +0.000 |
| Acerto de ausência | 0.500 | 0.500 | 0.500 | +0.000 |

## Split development

| Métrica | Lexical | Semântico | Híbrido | Híbrido − Lexical |
| --- | ---: | ---: | ---: | ---: |
| Precision@1 | 0.882 | 0.824 | 0.882 | +0.000 |
| Precision@3 | 0.745 | 0.745 | 0.765 | +0.020 |
| Precision@5 | 0.612 | 0.635 | 0.647 | +0.035 |
| Recall@1 | 0.281 | 0.222 | 0.281 | +0.000 |
| Recall@3 | 0.579 | 0.549 | 0.608 | +0.029 |
| Recall@5 | 0.743 | 0.705 | 0.764 | +0.021 |
| MRR | 0.917 | 0.850 | 0.922 | +0.005 |
| Hit Rate@5 | 1.000 | 0.882 | 1.000 | +0.000 |
| Acerto de ausência | 0.667 | 0.667 | 0.667 | +0.000 |

## Split test

| Métrica | Lexical | Semântico | Híbrido | Híbrido − Lexical |
| --- | ---: | ---: | ---: | ---: |
| Precision@1 | 1.000 | 1.000 | 1.000 | +0.000 |
| Precision@3 | 0.778 | 0.741 | 0.778 | +0.000 |
| Precision@5 | 0.644 | 0.622 | 0.644 | +0.000 |
| Recall@1 | 0.388 | 0.388 | 0.388 | +0.000 |
| Recall@3 | 0.739 | 0.683 | 0.739 | +0.000 |
| Recall@5 | 0.895 | 0.858 | 0.895 | +0.000 |
| MRR | 1.000 | 1.000 | 1.000 | +0.000 |
| Hit Rate@5 | 1.000 | 1.000 | 1.000 | +0.000 |
| Acerto de ausência | 0.333 | 0.333 | 0.333 | +0.000 |

## Resultados por categoria

| Categoria | MRR lexical | MRR semântico | MRR híbrido | Híbrido − lexical |
| --- | ---: | ---: | ---: | ---: |
| accent_variation | 1.000 | 1.000 | 1.000 | +0.000 |
| ambiguous | 1.000 | 1.000 | 1.000 | +0.000 |
| case_variation | 1.000 | 1.000 | 1.000 | +0.000 |
| implicit_intent | 1.000 | 1.000 | 1.000 | +0.000 |
| lexical_challenge | 1.000 | 1.000 | 1.000 | +0.000 |
| literal | 1.000 | 0.700 | 1.000 | +0.000 |
| multiple_constraints | 1.000 | 1.000 | 1.000 | +0.000 |
| no_result | — | — | — | — |
| numeric_filter | 1.000 | 1.000 | 1.000 | +0.000 |
| paraphrase | 1.000 | 1.000 | 1.000 | +0.000 |
| specific_feature | 0.778 | 0.714 | 0.778 | +0.000 |
| synonym | 0.812 | 0.800 | 0.833 | +0.021 |

## Diferenças do híbrido em relação ao lexical

- melhorias: ['QUERY-004', 'QUERY-013', 'QUERY-014'];
- regressões: ['nenhuma'];
- empates: 29 consultas.

## Exemplos de contribuições diferentes

- `QUERY-004`: `IMOB-027` não apareceu no ranking lexical, mas entrou pela posição semântica 2, com contribuição lexical zero;
- `QUERY-013`: `IMOB-029` tinha posições lexical 4 e semântica 5; a combinação o levou à posição híbrida 3;
- `QUERY-014`: `IMOB-048` não apareceu lexicalmente, mas entrou pela posição semântica 1, também com contribuição lexical zero.

## Conclusão e limitações

No split test, o híbrido empatou com o lexical em todas as métricas; portanto, não demonstrou melhora fora do desenvolvimento.

A conclusão principal deve considerar o split test. O catálogo e o teste são pequenos e sintéticos; diferenças modestas não demonstram generalização. O score RRF não é probabilidade. A política de ausência possui alta incerteza por haver somente três casos em development.
