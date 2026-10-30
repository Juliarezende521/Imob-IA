# Avaliação do classificador Transformer

## Seleção em validação

| Estratégia | Melhor epoch | Accuracy | Precision macro | Recall macro | F1 macro | Menor F1 | Tempo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Encoder congelado | 2 | 0,359 | 0,456 | 0,359 | 0,290 | 0,000 | 72,76 s |
| Fine-tuning completo | 4 | 0,531 | 0,645 | 0,531 | 0,514 | 0,375 | 326,40 s |

O fine-tuning completo venceu pelo critério primário, F1 macro. Mesmo assim,
ficou abaixo do baseline tradicional de validação (`0,764`). A configuração
foi congelada antes de qualquer avaliação no teste ou challenge.

Configuração congelada:

- SHA-256: `c484c07dcb34c8ff31e15404b19fe004032e54c4869bc5d6b34feec4a043cdf1`;
- estratégia: fine-tuning completo;
- epochs: 4;
- learning rate: `2e-5`;
- batch: 8;
- max length: 64;
- seed: 42;
- dispositivo: CPU.

## Teste antigo — comparação preliminar

O teste já havia sido observado no marco anterior, portanto não é o holdout
final.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | F1 weighted | Acertos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LogReg | 0,781 | 0,823 | 0,781 | 0,786 | 0,786 | 50/64 |
| Transformer | 0,344 | 0,365 | 0,344 | 0,334 | 0,334 | 22/64 |

Diferença de F1 macro: `−0,452`. O baseline acertou sozinho 31 casos, o
Transformer acertou sozinho 3, ambos acertaram 19 e ambos erraram 11.

Matriz do Transformer no teste, na ordem canônica dos oito rótulos:

```text
[2, 0, 1, 0, 0, 1, 3, 1]
[0, 4, 0, 3, 0, 0, 1, 0]
[0, 0, 5, 1, 0, 0, 2, 0]
[0, 0, 2, 2, 0, 0, 2, 2]
[0, 1, 2, 1, 0, 3, 1, 0]
[1, 1, 0, 0, 0, 3, 3, 0]
[0, 1, 0, 0, 0, 4, 3, 0]
[0, 0, 0, 1, 0, 0, 4, 3]
```

## Challenge final — primeira e única avaliação

Antes da execução foram validados o manifesto, o hash do challenge, o hash da
configuração e o hash do artefato. O receipt impede uma segunda execução.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | F1 weighted | Acertos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LogReg | 0,682 | 0,694 | 0,682 | 0,673 | 0,673 | 131/192 |
| Transformer | 0,438 | 0,462 | 0,438 | 0,385 | 0,385 | 84/192 |

Diferença de F1 macro: `−0,2875`. O baseline acertou sozinho 62 mensagens, o
Transformer acertou sozinho 15, ambos acertaram 69 e ambos erraram 46.

### F1 por classe

| Classe | Baseline | Transformer | Diferença |
| --- | ---: | ---: | ---: |
| `BUSCA_IMOVEL` | 0,681 | 0,074 | −0,607 |
| `AGENDAR_VISITA` | 0,698 | 0,324 | −0,373 |
| `DUVIDA_IMOVEL` | 0,711 | 0,571 | −0,140 |
| `DOCUMENTACAO_LOCACAO` | 0,655 | 0,458 | −0,197 |
| `FINANCEIRO_LOCACAO` | 0,667 | 0,148 | −0,519 |
| `MANUTENCAO_IMOVEL` | 0,792 | 0,727 | −0,065 |
| `ATENDIMENTO_HUMANO` | 0,780 | 0,488 | −0,291 |
| `FORA_DE_ESCOPO` | 0,400 | 0,292 | −0,108 |

O Transformer não melhorou nenhuma classe. A menor regressão ocorreu em
manutenção; as maiores, em busca e financeiro.

### Matrizes de confusão

Baseline:

```text
[16, 0, 1, 1, 0, 3, 2, 1]
[3, 15, 2, 1, 0, 0, 1, 2]
[1, 1, 16, 2, 0, 2, 1, 1]
[1, 2, 0, 19, 1, 0, 0, 1]
[0, 0, 1, 7, 13, 0, 0, 3]
[0, 0, 1, 2, 0, 21, 0, 0]
[0, 0, 0, 1, 0, 0, 23, 0]
[2, 1, 0, 1, 1, 3, 8, 8]
```

Transformer:

```text
[1, 3, 6, 0, 0, 2, 12, 0]
[0, 6, 2, 0, 0, 2, 6, 8]
[1, 1, 16, 2, 0, 0, 3, 1]
[0, 0, 3, 11, 1, 0, 6, 3]
[0, 1, 4, 9, 2, 3, 3, 2]
[0, 0, 0, 2, 0, 20, 1, 1]
[0, 1, 0, 0, 0, 0, 21, 2]
[1, 1, 1, 0, 0, 4, 10, 7]
```

Exemplos ilustrativos: o Transformer confundiu busca acessível com manutenção,
diversas demandas financeiras com documentação e assuntos fora de escopo com
atendimento humano ou manutenção. Os scores eram frequentemente próximos e não
devem ser interpretados como certeza. Todos os 108 erros permanecem registrados
no receipt, sem alteração dos dados.

Com apenas 192 mensagens sintéticas, a diferença é descritiva e possui
incerteza amostral. Não foi executado bootstrap; não se afirma significância
estatística. Ainda assim, o resultado observado não sustenta superioridade do
Transformer.

## Benchmark operacional em CPU

| Medida | Baseline | Transformer |
| --- | ---: | ---: |
| Treino final | 0,169 s | 328,05 s |
| Tamanho | 424.637 bytes | 545.253.407 bytes |
| Latência unitária mediana | 3,95 ms | 41,47 ms |
| Latência unitária p95 | 5,09 ms | 49,77 ms |
| Lote de 8, mediana | 3,26 ms | 169,45 ms |
| Throughput do lote | 2.453 msg/s | 47,2 msg/s |
| Carregamento | não aplicável | 363,64 ms |

Foram usadas 30 repetições unitárias, 20 em lote e três carregamentos, após um
warmup unitário e um em lote. RAM não foi medida com precisão.

## Conclusão

O baseline tradicional é melhor em qualidade, latência, tamanho, custo e
manutenção neste experimento. O Transformer deve permanecer como experimento e
não substituir o baseline. O resultado provavelmente reflete o volume reduzido,
a natureza sintética e a mudança de famílias entre splits. Nenhum ajuste foi
feito depois da abertura do challenge.
