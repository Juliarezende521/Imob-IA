# Avaliação da busca semântica

## Configuração

- modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- dimensão: 384;
- licença do modelo: Apache-2.0;
- sentence-transformers: 5.1.2;
- similaridade: cosseno exato em memória;
- limiar calibrado em development: 0.200.

## Comparação geral

| Métrica | Lexical | Semântico | Diferença |
| --- | ---: | ---: | ---: |
| Precision@1 | 0.923 | 0.885 | -0.038 |
| Precision@3 | 0.756 | 0.744 | -0.013 |
| Precision@5 | 0.623 | 0.631 | +0.008 |
| Recall@1 | 0.318 | 0.280 | -0.038 |
| Recall@3 | 0.634 | 0.596 | -0.038 |
| Recall@5 | 0.796 | 0.758 | -0.038 |
| MRR | 0.946 | 0.902 | -0.044 |
| Hit Rate@5 | 1.000 | 0.923 | -0.077 |
| Acerto de ausência | 0.500 | 0.500 | +0.000 |

## Split development

| Métrica | Lexical | Semântico | Diferença |
| --- | ---: | ---: | ---: |
| Precision@1 | 0.882 | 0.824 | -0.059 |
| Precision@3 | 0.745 | 0.745 | +0.000 |
| Precision@5 | 0.612 | 0.635 | +0.024 |
| Recall@1 | 0.281 | 0.222 | -0.059 |
| Recall@3 | 0.579 | 0.549 | -0.029 |
| Recall@5 | 0.743 | 0.705 | -0.038 |
| MRR | 0.917 | 0.850 | -0.067 |
| Hit Rate@5 | 1.000 | 0.882 | -0.118 |
| Acerto de ausência | 0.667 | 0.667 | +0.000 |

## Split test

| Métrica | Lexical | Semântico | Diferença |
| --- | ---: | ---: | ---: |
| Precision@1 | 1.000 | 1.000 | +0.000 |
| Precision@3 | 0.778 | 0.741 | -0.037 |
| Precision@5 | 0.644 | 0.622 | -0.022 |
| Recall@1 | 0.388 | 0.388 | +0.000 |
| Recall@3 | 0.739 | 0.683 | -0.056 |
| Recall@5 | 0.895 | 0.858 | -0.037 |
| MRR | 1.000 | 1.000 | +0.000 |
| Hit Rate@5 | 1.000 | 1.000 | +0.000 |
| Acerto de ausência | 0.333 | 0.333 | +0.000 |

## Resultados por categoria

| Categoria | MRR lexical | MRR semântico | Diferença |
| --- | ---: | ---: | ---: |
| accent_variation | 1.000 | 1.000 | +0.000 |
| ambiguous | 1.000 | 1.000 | +0.000 |
| case_variation | 1.000 | 1.000 | +0.000 |
| implicit_intent | 1.000 | 1.000 | +0.000 |
| lexical_challenge | 1.000 | 1.000 | +0.000 |
| literal | 1.000 | 0.700 | -0.300 |
| multiple_constraints | 1.000 | 1.000 | +0.000 |
| no_result | — | — | — |
| numeric_filter | 1.000 | 1.000 | +0.000 |
| paraphrase | 1.000 | 1.000 | +0.000 |
| specific_feature | 0.778 | 0.714 | -0.063 |
| synonym | 0.812 | 0.800 | -0.012 |

## Diferenças por consulta

- melhorias: ['QUERY-004', 'QUERY-010', 'QUERY-014', 'QUERY-017'];
- regressões: ['QUERY-008', 'QUERY-009', 'QUERY-013', 'QUERY-021'];
- empates: 24 consultas.

## Interpretação e limitações

A comparação usa apenas 32 consultas e 48 imóveis sintéticos. O split test é pequeno; diferenças são preliminares. Similaridade não é probabilidade. O limiar foi escolhido somente em development e pode não generalizar. Não há busca híbrida, índice vetorial ou RAG.
