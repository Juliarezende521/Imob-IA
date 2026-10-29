# Busca híbrida com Weighted RRF

## Estratégia

A busca combina posições lexicais e semânticas, não seus scores incompatíveis:

```text
score = peso_lexical / (k + posição_lexical)
      + peso_semântico / (k + posição_semântica)
```

Um candidato ausente de um ranking não recebe posição artificial: sua
contribuição naquele método é zero. Os campos de explicabilidade distinguem
posição ausente, score original e contribuição RRF.

## Protocolo congelado antes do teste

- filtros estruturados executados antes dos dois rankings;
- todos os candidatos filtrados participam do ranking semântico;
- união completa dos rankings;
- `k=60`, definido antes da calibração;
- pesos lexicais de 0,0 a 1,0, em passos de 0,1;
- extremos usados somente como controles;
- seleção por MRR, Recall@5, Precision@5, Hit@5, ausência e equilíbrio;
- evidência mínima: presença lexical ou similaridade semântica de 0,20;
- desempate: score híbrido, score semântico, score lexical e ID.

Somente as 20 consultas de `development` participaram da escolha. A configuração
final é peso lexical `0,9` e semântico `0,1`. O split `test` foi executado apenas
depois desse congelamento.

## Execução

```bash
python -m imob_ai.evaluation.runner --engine hybrid
python -m imob_ai.evaluation.runner --engine compare \
  --output docs/hybrid-search-evaluation.md
```

## Limitações

- o catálogo e a grade de desenvolvimento são pequenos;
- RRF usa posições e descarta a magnitude dos scores;
- `k=60` não foi otimizado, por decisão de simplicidade;
- a política de ausência tem somente três exemplos em desenvolvimento;
- peso 0,9 mostra que o sinal lexical domina neste conjunto;
- o score híbrido não é probabilidade ou confiança.
