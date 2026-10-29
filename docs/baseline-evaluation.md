# Avaliação do baseline lexical

Este relatório mede o baseline determinístico sobre julgamentos manuais.
Não há embeddings ou modelo de Inteligência Artificial nesta etapa.

## Resumo

| Grupo | Casos | Recuperação | Ausência | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | Hit@5 | Acc. ausência |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Geral | 32 | 26 | 6 | 0.923 | 0.756 | 0.623 | 0.318 | 0.634 | 0.796 | 0.946 | 1.000 | 0.500 |
| development | 20 | 17 | 3 | 0.882 | 0.745 | 0.612 | 0.281 | 0.579 | 0.743 | 0.917 | 1.000 | 0.667 |
| test | 12 | 9 | 3 | 1.000 | 0.778 | 0.644 | 0.388 | 0.739 | 0.895 | 1.000 | 1.000 | 0.333 |

## Resultados por categoria

| Grupo | Casos | Recuperação | Ausência | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | Hit@5 | Acc. ausência |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accent_variation | 2 | 2 | 0 | 1.000 | 0.667 | 0.400 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | — |
| ambiguous | 1 | 1 | 0 | 1.000 | 1.000 | 0.600 | 0.053 | 0.158 | 0.158 | 1.000 | 1.000 | — |
| case_variation | 2 | 2 | 0 | 1.000 | 1.000 | 0.800 | 0.250 | 0.750 | 1.000 | 1.000 | 1.000 | — |
| implicit_intent | 3 | 3 | 0 | 1.000 | 0.778 | 0.600 | 0.454 | 0.694 | 0.769 | 1.000 | 1.000 | — |
| lexical_challenge | 2 | 2 | 0 | 1.000 | 1.000 | 1.000 | 0.133 | 0.400 | 0.667 | 1.000 | 1.000 | — |
| literal | 3 | 3 | 0 | 1.000 | 0.667 | 0.467 | 0.556 | 0.889 | 1.000 | 1.000 | 1.000 | — |
| multiple_constraints | 2 | 2 | 0 | 1.000 | 0.667 | 0.400 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | — |
| no_result | 6 | 0 | 6 | — | — | — | — | — | — | — | — | 0.500 |
| numeric_filter | 1 | 1 | 0 | 1.000 | 1.000 | 0.800 | 0.250 | 0.750 | 1.000 | 1.000 | 1.000 | — |
| paraphrase | 3 | 3 | 0 | 1.000 | 0.778 | 0.733 | 0.267 | 0.467 | 0.667 | 1.000 | 1.000 | — |
| specific_feature | 3 | 3 | 0 | 0.667 | 0.444 | 0.333 | 0.333 | 0.667 | 0.833 | 0.778 | 1.000 | — |
| synonym | 4 | 4 | 0 | 0.750 | 0.750 | 0.800 | 0.094 | 0.283 | 0.597 | 0.812 | 1.000 | — |

## Melhores resultados

- `QUERY-001`: MRR = 1.000, Recall@5 = 1.000; recuperados = ['IMOB-001', 'IMOB-025', 'IMOB-045', 'IMOB-017', 'IMOB-033'].
- `QUERY-006`: MRR = 1.000, Recall@5 = 1.000; recuperados = ['IMOB-012', 'IMOB-002', 'IMOB-001', 'IMOB-010'].
- `QUERY-009`: MRR = 1.000, Recall@5 = 1.000; recuperados = ['IMOB-017', 'IMOB-012', 'IMOB-001', 'IMOB-010', 'IMOB-036'].
- `QUERY-011`: MRR = 1.000, Recall@5 = 1.000; recuperados = ['IMOB-010', 'IMOB-034'].
- `QUERY-012`: MRR = 1.000, Recall@5 = 1.000; recuperados = ['IMOB-019', 'IMOB-043', 'IMOB-007', 'IMOB-031'].

## Piores resultados

- `QUERY-013`: MRR = 0.250, Recall@5 = 0.500; recuperados = ['IMOB-042', 'IMOB-032', 'IMOB-048', 'IMOB-029', 'IMOB-002'].
- `QUERY-017`: MRR = 0.333, Recall@5 = 1.000; recuperados = ['IMOB-047', 'IMOB-003', 'IMOB-019', 'IMOB-027', 'IMOB-023'].
- `QUERY-014`: MRR = 1.000, Recall@5 = 0.158; recuperados = ['IMOB-042', 'IMOB-008', 'IMOB-047'].
- `QUERY-029`: MRR = 1.000, Recall@5 = 0.500; recuperados = ['IMOB-044', 'IMOB-008', 'IMOB-040', 'IMOB-012', 'IMOB-020'].
- `QUERY-010`: MRR = 1.000, Recall@5 = 0.500; recuperados = ['IMOB-002', 'IMOB-018', 'IMOB-006', 'IMOB-010', 'IMOB-026'].

## Falhas individuais

- `QUERY-018`: ausência correta = `False`; recuperados = ['IMOB-001', 'IMOB-017', 'IMOB-025', 'IMOB-033', 'IMOB-005', 'IMOB-013', 'IMOB-021', 'IMOB-037', 'IMOB-045', 'IMOB-029'].
- `QUERY-031`: ausência correta = `False`; recuperados = ['IMOB-017'].
- `QUERY-032`: ausência correta = `False`; recuperados = ['IMOB-017', 'IMOB-014', 'IMOB-002', 'IMOB-012', 'IMOB-001', 'IMOB-010', 'IMOB-020', 'IMOB-018', 'IMOB-011', 'IMOB-026', 'IMOB-036', 'IMOB-025', 'IMOB-034', 'IMOB-044', 'IMOB-027', 'IMOB-033', 'IMOB-042', 'IMOB-016', 'IMOB-006', 'IMOB-005', 'IMOB-007', 'IMOB-013', 'IMOB-030', 'IMOB-021', 'IMOB-038', 'IMOB-023', 'IMOB-031', 'IMOB-037', 'IMOB-039', 'IMOB-045', 'IMOB-047'].

## Fórmulas e denominadores

- `Precision@K = relevantes recuperados nas K primeiras posições / K`.
- `Recall@K = relevantes recuperados nas K primeiras posições / total de relevantes`.
- `MRR = média de 1 / posição do primeiro relevante`.
- `Hit Rate@K = consultas com ao menos um relevante no top K / consultas de recuperação`.
- `No-result accuracy = consultas corretamente vazias / consultas sem resultado esperado`.
- Consultas sem resultado esperado não entram nas médias de Precision, Recall, MRR ou Hit Rate.

## Regra contra vazamento

O split de desenvolvimento pode orientar ajustes futuros. O split de teste deve permanecer estável e não deve ser consultado repetidamente para ajustar implementações. Mudanças posteriores nos julgamentos precisam de justificativa e versionamento.

## Limitações

- catálogo pequeno e totalmente sintético;
- julgamentos manuais e parcialmente subjetivos;
- poucas consultas por categoria;
- ausência de avaliadores independentes;
- métricas válidas para comparação interna, não para generalização ao mercado real.
