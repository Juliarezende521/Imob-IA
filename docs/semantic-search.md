# Busca semântica local

## Arquitetura

A consulta passa primeiro pelos mesmos filtros estruturados do baseline. Apenas
os candidatos válidos são transformados em documentos, codificados e ordenados
por similaridade cosseno. Imóveis indisponíveis continuam excluídos por padrão.

O serviço usa busca exata em memória. Com apenas 48 imóveis, FAISS, Chroma ou
outro banco vetorial adicionaria complexidade sem benefício mensurável.

## Modelo

- modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- biblioteca: `sentence-transformers==5.1.2`;
- dimensão: 384;
- licença indicada no model card: Apache 2.0;
- execução: local, sem API paga.

Fontes: [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
e [pacote oficial](https://pypi.org/project/sentence-transformers/5.1.2/).

O modelo é carregado somente no primeiro uso. A primeira execução baixa os
pesos; as seguintes usam o cache local do Hugging Face, fora do Git.

## Documento canônico

Cada imóvel é representado sempre na mesma ordem: finalidade, tipo, cidade,
bairro, preço, área, quantidades, aceitação de animais, mobília,
características e descrição. O ID não participa do texto para não virar sinal
semântico. Disponibilidade também não participa porque é filtro operacional.

## Similaridade e cache

Os embeddings são normalizados pelo encoder, e o serviço calcula cosseno de
forma explícita. O score varia matematicamente entre -1 e 1, mas não é
probabilidade nem confiança.

Documentos são armazenados apenas em memória, com chave formada pelo nome do
encoder e pelo texto canônico. O cache vive durante a instância do serviço e é
invalidado ao encerrá-la ou quando modelo/texto muda. Nada é persistido no Git.

## Limiar

Foram avaliados somente nas 20 consultas de desenvolvimento:

| Limiar | Recall@5 | Acerto de ausência | Equilíbrio harmônico |
| ---: | ---: | ---: | ---: |
| 0,00 | 0,705 | 0,667 | 0,685 |
| 0,05 | 0,705 | 0,667 | 0,685 |
| 0,10 | 0,705 | 0,667 | 0,685 |
| 0,15 | 0,705 | 0,667 | 0,685 |
| 0,20 | 0,705 | 0,667 | 0,685 |
| 0,25 | 0,690 | 0,667 | 0,678 |
| 0,30 | 0,587 | 0,667 | 0,625 |
| 0,35 | 0,587 | 0,667 | 0,625 |
| 0,40 | 0,511 | 0,667 | 0,578 |
| 0,45 | 0,466 | 0,667 | 0,549 |
| 0,50 | 0,411 | 1,000 | 0,583 |
| 0,55 | 0,239 | 1,000 | 0,386 |
| 0,60 | 0,059 | 1,000 | 0,111 |
| 0,65 | 0,059 | 1,000 | 0,111 |
| 0,70 | 0,000 | 1,000 | 0,000 |
| 0,75 | 0,000 | 1,000 | 0,000 |
| 0,80 | 0,000 | 1,000 | 0,000 |

O limiar congelado é `0,20`. Os valores de `0,00` a `0,20` empataram; o maior
foi escolhido como desempate conservador, pois rejeita scores mais baixos sem
perder recuperação observada em desenvolvimento. O teste não participou da
escolha. A amostra é pequena, portanto o valor não deve ser tratado como
universal. O serviço aceita `threshold=None` para inspeção do ranking bruto.

## Execução

```bash
# Testes comuns, sem download
pytest -m "not model_integration"

# Teste real opcional
RUN_MODEL_INTEGRATION=1 pytest -m model_integration

# Baseline preservado
python -m imob_ai.evaluation.runner --engine lexical

# Avaliação semântica
python -m imob_ai.evaluation.runner --engine semantic

# Comparação
python -m imob_ai.evaluation.runner --engine compare \
  --output docs/semantic-search-evaluation.md
```

## Limitações

- similaridade pode aproximar textos irrelevantes;
- o limiar foi calibrado em poucos exemplos sintéticos;
- números dentro do documento não garantem raciocínio numérico do modelo;
- scores de consultas diferentes não são diretamente comparáveis;
- não há combinação com score lexical, índice vetorial, fine-tuning ou RAG.

Busca lexical compara termos; busca semântica compara vetores; busca híbrida
combinaria sinais, mas não faz parte deste marco.
