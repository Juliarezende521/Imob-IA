# Avaliação do classificador de intenções

O dataset é inteiramente sintético. O teste foi avaliado somente após o congelamento do pipeline em validação.

## Configurações avaliadas em validação

| Configuração | Accuracy | Precision macro | Recall macro | F1 macro | Menor F1 | Dimensões |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| word_unigram | 0.656 | 0.678 | 0.656 | 0.660 | 0.400 | 543 |
| word_bigram | 0.688 | 0.702 | 0.688 | 0.688 | 0.421 | 1563 |
| char_wb | 0.750 | 0.771 | 0.750 | 0.746 | 0.500 | 2624 |
| word_char | 0.766 | 0.780 | 0.766 | 0.764 | 0.462 | 4187 |

## Pipeline congelado

- configuração: `word_char`;
- tipo: `combined`;
- n-gramas principais: `(1, 2)`;
- C: `1.0`;
- classificador: Regressão Logística;
- seed: `42`;
- scikit-learn: `1.6.1`.

## Métricas agregadas

| Conjunto | Modelo | Accuracy | Precision macro | Recall macro | F1 macro | F1 weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validação | Dummy | 0.125 | 0.016 | 0.125 | 0.028 | 0.028 |
| Validação | TF-IDF + LogReg | 0.766 | 0.780 | 0.766 | 0.764 | 0.764 |
| Teste | Dummy | 0.125 | 0.016 | 0.125 | 0.028 | 0.028 |
| Teste | TF-IDF + LogReg | 0.781 | 0.823 | 0.781 | 0.786 | 0.786 |

## Métricas por classe no teste

| Classe | Precision | Recall | F1 | Suporte |
| --- | ---: | ---: | ---: | ---: |
| BUSCA_IMOVEL | 0.857 | 0.750 | 0.800 | 8 |
| AGENDAR_VISITA | 0.700 | 0.875 | 0.778 | 8 |
| DUVIDA_IMOVEL | 1.000 | 0.875 | 0.933 | 8 |
| DOCUMENTACAO_LOCACAO | 0.857 | 0.750 | 0.800 | 8 |
| FINANCEIRO_LOCACAO | 1.000 | 1.000 | 1.000 | 8 |
| MANUTENCAO_IMOVEL | 1.000 | 0.625 | 0.769 | 8 |
| ATENDIMENTO_HUMANO | 0.500 | 0.875 | 0.636 | 8 |
| FORA_DE_ESCOPO | 0.667 | 0.500 | 0.571 | 8 |

## Matriz de confusão do teste

Ordem das classes: `BUSCA_IMOVEL`, `AGENDAR_VISITA`, `DUVIDA_IMOVEL`, `DOCUMENTACAO_LOCACAO`, `FINANCEIRO_LOCACAO`, `MANUTENCAO_IMOVEL`, `ATENDIMENTO_HUMANO`, `FORA_DE_ESCOPO`.

```text
[6, 1, 0, 0, 0, 0, 1, 0]
[0, 7, 0, 1, 0, 0, 0, 0]
[1, 0, 7, 0, 0, 0, 0, 0]
[0, 1, 0, 6, 0, 0, 0, 1]
[0, 0, 0, 0, 8, 0, 0, 0]
[0, 0, 0, 0, 0, 5, 2, 1]
[0, 1, 0, 0, 0, 0, 7, 0]
[0, 0, 0, 0, 0, 0, 4, 4]
```

## Confusões mais frequentes

- FORA_DE_ESCOPO → ATENDIMENTO_HUMANO: 4 caso(s).
- MANUTENCAO_IMOVEL → ATENDIMENTO_HUMANO: 2 caso(s).
- BUSCA_IMOVEL → AGENDAR_VISITA: 1 caso(s).
- BUSCA_IMOVEL → ATENDIMENTO_HUMANO: 1 caso(s).
- AGENDAR_VISITA → DOCUMENTACAO_LOCACAO: 1 caso(s).
- DUVIDA_IMOVEL → BUSCA_IMOVEL: 1 caso(s).
- DOCUMENTACAO_LOCACAO → FORA_DE_ESCOPO: 1 caso(s).
- DOCUMENTACAO_LOCACAO → AGENDAR_VISITA: 1 caso(s).
- MANUTENCAO_IMOVEL → FORA_DE_ESCOPO: 1 caso(s).
- ATENDIMENTO_HUMANO → AGENDAR_VISITA: 1 caso(s).

## Erros individuais

- `INTENT-0042`: “Quero conhecer as opções da imobiliária” — esperado `BUSCA_IMOVEL`, previsto `AGENDAR_VISITA`; scores: AGENDAR_VISITA=0.291, BUSCA_IMOVEL=0.229, ATENDIMENTO_HUMANO=0.103; família `busca_imovel__search_open_ended`; dificuldade `easy`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0046`: “ME MOSTRA O QUE ESTÁ DISPONÍVEL” — esperado `BUSCA_IMOVEL`, previsto `ATENDIMENTO_HUMANO`; scores: ATENDIMENTO_HUMANO=0.191, BUSCA_IMOVEL=0.190, MANUTENCAO_IMOVEL=0.142; família `busca_imovel__search_open_ended`; dificuldade `medium`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0093`: “Preciso saber se o agendamento continua válido” — esperado `AGENDAR_VISITA`, previsto `DOCUMENTACAO_LOCACAO`; scores: DOCUMENTACAO_LOCACAO=0.174, FINANCEIRO_LOCACAO=0.154, AGENDAR_VISITA=0.151; família `agendar_visita__visit_confirm`; dificuldade `medium`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0143`: “Não estou buscando outras opções, só quero confirmar esta” — esperado `DUVIDA_IMOVEL`, previsto `BUSCA_IMOVEL`; scores: BUSCA_IMOVEL=0.232, FORA_DE_ESCOPO=0.198, FINANCEIRO_LOCACAO=0.133; família `duvida_imovel__question_availability`; dificuldade `hard`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0188`: “q falta pra liberar as chaves?” — esperado `DOCUMENTACAO_LOCACAO`, previsto `FORA_DE_ESCOPO`; scores: FORA_DE_ESCOPO=0.152, DOCUMENTACAO_LOCACAO=0.146, ATENDIMENTO_HUMANO=0.135; família `documentacao_locacao__docs_keys_requirements`; dificuldade `medium`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0191`: “Não estou marcando visita; quero entender a liberação documental” — esperado `DOCUMENTACAO_LOCACAO`, previsto `AGENDAR_VISITA`; scores: AGENDAR_VISITA=0.228, FINANCEIRO_LOCACAO=0.156, DOCUMENTACAO_LOCACAO=0.148; família `documentacao_locacao__docs_keys_requirements`; dificuldade `hard`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0284`: “manutencao urgente por favor” — esperado `MANUTENCAO_IMOVEL`, previsto `FORA_DE_ESCOPO`; scores: FORA_DE_ESCOPO=0.163, ATENDIMENTO_HUMANO=0.161, MANUTENCAO_IMOVEL=0.150; família `manutencao_imovel__maintenance_urgent`; dificuldade `medium`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0287`: “Quero falar com alguém porque o vazamento está aumentando” — esperado `MANUTENCAO_IMOVEL`, previsto `ATENDIMENTO_HUMANO`; scores: ATENDIMENTO_HUMANO=0.543, MANUTENCAO_IMOVEL=0.132, BUSCA_IMOVEL=0.075; família `manutencao_imovel__maintenance_urgent`; dificuldade `hard`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0288`: “A prioridade é resolver o defeito, mesmo que um atendente retorne” — esperado `MANUTENCAO_IMOVEL`, previsto `ATENDIMENTO_HUMANO`; scores: ATENDIMENTO_HUMANO=0.478, MANUTENCAO_IMOVEL=0.119, FORA_DE_ESCOPO=0.090; família `manutencao_imovel__maintenance_urgent`; dificuldade `hard`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0335`: “Não quero cancelar visita; quero apenas sair do atendimento automático” — esperado `ATENDIMENTO_HUMANO`, previsto `AGENDAR_VISITA`; scores: AGENDAR_VISITA=0.346, ATENDIMENTO_HUMANO=0.235, FINANCEIRO_LOCACAO=0.084; família `atendimento_humano__human_cancel_bot`; dificuldade `hard`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0377`: “Me indique um filme” — esperado `FORA_DE_ESCOPO`, previsto `ATENDIMENTO_HUMANO`; scores: ATENDIMENTO_HUMANO=0.287, FORA_DE_ESCOPO=0.171, MANUTENCAO_IMOVEL=0.139; família `fora_de_escopo__outside_entertainment`; dificuldade `easy`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0379`: “Quero ouvir uma música nova” — esperado `FORA_DE_ESCOPO`, previsto `ATENDIMENTO_HUMANO`; scores: ATENDIMENTO_HUMANO=0.167, AGENDAR_VISITA=0.163, FINANCEIRO_LOCACAO=0.155; família `fora_de_escopo__outside_entertainment`; dificuldade `easy`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0381`: “Preciso de uma sugestão de livro” — esperado `FORA_DE_ESCOPO`, previsto `ATENDIMENTO_HUMANO`; scores: ATENDIMENTO_HUMANO=0.218, FORA_DE_ESCOPO=0.173, MANUTENCAO_IMOVEL=0.158; família `fora_de_escopo__outside_entertainment`; dificuldade `medium`. Possível causa: vocabulário compartilhado e cenário não observado no treino.
- `INTENT-0383`: “Não quero falar de locação, só escolher algo para assistir” — esperado `FORA_DE_ESCOPO`, previsto `ATENDIMENTO_HUMANO`; scores: ATENDIMENTO_HUMANO=0.244, BUSCA_IMOVEL=0.172, DOCUMENTACAO_LOCACAO=0.128; família `fora_de_escopo__outside_entertainment`; dificuldade `hard`. Possível causa: vocabulário compartilhado e cenário não observado no treino.

## Limitações

- mensagens sintéticas não representam toda a linguagem real;
- apenas uma família por classe existe em validação e teste;
- scores de `predict_proba` não são certeza e não foram calibrados;
- erros ortográficos e ambiguidades reais podem ser mais variados;
- o classificador não está pronto para produção.
