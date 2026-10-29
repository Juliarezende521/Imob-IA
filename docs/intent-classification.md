# Classificação de intenções

## Escopo

Classificação escolhe um rótulo para uma mensagem; recuperação encontra itens
relevantes em um catálogo. Neste marco, o classificador não executa busca nem
produz respostas. Ele identifica somente a intenção principal.

## Taxonomia

| Intenção | Descrição e exemplo positivo | Não pertence / fronteira |
| --- | --- | --- |
| `BUSCA_IMOVEL` | pessoa ainda procura opções. Ex.: “Procuro casa para alugar” | “Essa casa aceita pet?” já trata de uma unidade e é `DUVIDA_IMOVEL` |
| `AGENDAR_VISITA` | marcar, remarcar, cancelar ou confirmar. Ex.: “Posso conhecer amanhã?” | perguntar sobre cômodos é `DUVIDA_IMOVEL`; cancelar visita continua sendo agendamento |
| `DUVIDA_IMOVEL` | pergunta sobre unidade conhecida. Ex.: “Esse apartamento tem garagem?” | “Quero apartamento com garagem” ainda é `BUSCA_IMOVEL` |
| `DOCUMENTACAO_LOCACAO` | cadastro, documentos e garantias. Ex.: “Preciso de fiador?” | boleto já emitido pertence a `FINANCEIRO_LOCACAO` |
| `FINANCEIRO_LOCACAO` | aluguel, boleto, pagamento, vencimento, reajuste ou cobrança. Ex.: “Quero a segunda via” | comprovante exigido no cadastro é `DOCUMENTACAO_LOCACAO` |
| `MANUTENCAO_IMOVEL` | defeito, reparo ou assistência. Ex.: “A pia está vazando” | desejar imóvel reformado durante busca é `BUSCA_IMOVEL` |
| `ATENDIMENTO_HUMANO` | pedido explícito de pessoa. Ex.: “Quero falar com um atendente” | uma mensagem apenas difícil não recebe este rótulo |
| `FORA_DE_ESCOPO` | assunto alheio ao fluxo. Ex.: “Qual a previsão do tempo?” | demanda imobiliária rara não deve ser descartada como fora de escopo |

Casos ambíguos são resolvidos pela ação principal explicitamente solicitada.
“Quero falar com alguém porque há vazamento” é manutenção se resolver o defeito
for a prioridade, mas atendimento humano se o pedido central for sair do fluxo
automático. “Não quero visitar, só saber o preço” é dúvida sobre imóvel. Negar
uma ação não a torna a intenção principal. Mencionar imóvel também não transforma
automaticamente uma dúvida específica em busca.

## Dados e isolamento

O dataset é totalmente sintético e tem 384 mensagens balanceadas entre oito
classes. A divisão é 256/64/64 para treino/validação/teste. Ela foi feita
manualmente e de forma determinística por
`scenario_family`: todas as paráfrases de uma situação ficam no mesmo split.
Isso reduz vazamento por frases quase equivalentes. O teste permanece oculto
durante escolha de configuração. Não há sorteio da divisão; a seed 42 pertence
ao treinamento reproduzível do modelo.

## Modelos

O baseline ingênuo é `DummyClassifier(strategy="most_frequent")`. O baseline
aprendido combina TF-IDF e Regressão Logística multiclasse. TF-IDF representa a
importância relativa de palavras ou fragmentos; n-gramas preservam pequenos
padrões locais. Foram predefinidas quatro configurações: unigramas de palavras,
palavras de uma e duas unidades, `char_wb` e união de palavras com caracteres.

A seleção usa primeiro F1 macro na validação; em empate, menor F1 por classe,
recall macro, menor complexidade e menor dimensionalidade. Somente depois do
congelamento, o modelo é refeito com treino + validação e avaliado uma vez no
teste. A seed é 42 e a versão direta de scikit-learn é 1.6.1.

## Execução

```bash
python -m imob_ai.intent.dataset
python -m imob_ai.intent.train
python -m imob_ai.intent.evaluate
```

Os scores expostos pelo pipeline são estimativas relativas do classificador,
não probabilidades calibradas nem garantias de acerto.

Um Transformer poderá ser comparado em um marco futuro, idealmente depois de
ampliar e revisar dados. Aqui ele adicionaria custo e complexidade sem resolver
as limitações do conjunto sintético pequeno.
