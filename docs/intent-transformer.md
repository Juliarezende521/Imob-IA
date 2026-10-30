# Classificador de intenções com Transformer

## Escopo

Este marco compara o baseline TF-IDF + Regressão Logística com um Transformer
local para classificação de oito intenções. Ele não implementa chatbot, RAG,
API ou geração de respostas.

## Modelo e decisão de segurança

O modelo é `distilbert/distilbert-base-multilingual-cased`, DistilBERT
multilíngue cased com seis camadas, dimensão 768 e 12 cabeças. O model card
oficial declara 104 idiomas, aproximadamente 134 milhões de parâmetros e
licença Apache 2.0:
https://huggingface.co/distilbert/distilbert-base-multilingual-cased

A revisão inicialmente considerada foi
`6045845d9e2b056487062a98a902d8304d76441f`. Ela continha somente
`pytorch_model.bin`, bloqueado por Transformers 4.57.6 com PyTorch 2.2.2 devido
à proteção relacionada à CVE-2025-32434. Nenhum treinamento ou avaliação havia
ocorrido.

Após autorização explícita, foi adotada a revisão oficial
`45c032ab32cc946ad88a166f7cb282f58c753c2e`, que contém
`model.safetensors`. Todos os carregamentos remotos exigem
`use_safetensors=True` e `trust_remote_code=False`; não existe fallback para
`.bin`. O modelo final foi salvo com `safe_serialization=True`. PyTorch foi
mantido em 2.2.2.

## Ambiente

- Python 3.9.6;
- PyTorch 2.2.2;
- Transformers 4.57.6;
- Sentence Transformers 5.1.2;
- macOS 15.7.4, processo `x86_64`, 16 GiB de RAM;
- treinamento em CPU.

MPS passou em inferência simples, mas falhou no primeiro gradiente com erro
interno do compilador Metal (`ndArrayIdentity`). A tentativa encerrou antes de
uma epoch e não produziu configuração ou artefato. O experimento recomeçou em
CPU, sem alterar modelo, dados ou hiperparâmetros. RAM de treinamento não foi
medida com precisão.

## Rótulos

A ordem canônica congelada é:

1. `BUSCA_IMOVEL`;
2. `AGENDAR_VISITA`;
3. `DUVIDA_IMOVEL`;
4. `DOCUMENTACAO_LOCACAO`;
5. `FINANCEIRO_LOCACAO`;
6. `MANUTENCAO_IMOVEL`;
7. `ATENDIMENTO_HUMANO`;
8. `FORA_DE_ESCOPO`.

O hash SHA-256 da taxonomia é
`b6fe3207ad4959e8ade126081b80c7d47b778dd9c2c8e7140939d1d7d0f6c1b5`.
O mapping está salvo no código, na configuração congelada e no `config.json`
local do modelo.

## Tokenização

O `DistilBertTokenizerFast` preserva caixa. A análise usou somente os 320
registros de treino + validação: mínimo 6 tokens, mediana 13, percentil 95 igual
a 21 e máximo 24. Foi congelado `max_length=64`, com padding e truncamento;
nenhuma mensagem de desenvolvimento foi truncada. Texto vazio é rejeitado.

## Treinamento e reprodução

Foram predefinidas duas estratégias:

- encoder congelado, learning rate `5e-4`;
- fine-tuning completo, learning rate `2e-5`.

Ambas usaram batch 8, até cinco epochs, weight decay 0,01, warmup 0,1,
gradient clipping 1,0, seed 42 e early stopping com paciência 2. Gradientes
usaram somente treino; epoch e estratégia foram escolhidas somente pela
validação. Teste e challenge não participaram da seleção.

O fine-tuning completo venceu e a epoch 4 foi congelada. O treino final usou
treino + validação por quatro epochs, sem early stopping. Seeds controlam
Python, NumPy, PyTorch, DataLoader e inicialização da cabeça, mas determinismo
absoluto pode variar entre dispositivos.

Comandos separados:

```bash
python -m imob_ai.intent_transformer.train
python -m imob_ai.intent_transformer.evaluate
python -m imob_ai.intent_transformer.benchmark
python -m imob_ai.intent_transformer.evaluate_challenge --confirm-frozen-config
```

O último comando recusa execução sem confirmação, com hash divergente ou quando
o receipt já existe. Pesos e cache ficam em diretórios ignorados pelo Git. Não
houve upload ao Hub, API paga ou `push_to_hub`.

## Limitações

São apenas 256 mensagens de treino sintéticas, distribuídas em poucas famílias.
O modelo possui cerca de 135 milhões de parâmetros e pode não se adaptar bem a
esse volume. Scores softmax não são certeza nem foram calibrados. Resultados do
challenge não autorizam ajuste posterior e não comprovam desempenho em produção.
