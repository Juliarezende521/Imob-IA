# Challenge final de classificação de intenções

## Motivo e protocolo

O `challenge-v1` é um holdout final sintético criado e congelado antes do
desenvolvimento de qualquer Transformer. A validação orienta escolhas; o teste
existente já foi observado e serve como comparação preliminar; o challenge fica
intocado para uma única comparação final.

Protocolo futuro obrigatório:

1. desenvolver o Transformer usando treino;
2. escolher configurações usando validação;
3. congelar modelo, tokenizer, pré-processamento e hiperparâmetros;
4. avaliar no teste existente como comparação preliminar;
5. somente depois executar uma única avaliação no challenge;
6. nunca alterar o challenge após observar resultados.

Neste marco nenhum modelo, vetorizer, classificador, predição, inferência ou
métrica foi executado no challenge.

## Estrutura e distribuição

O arquivo possui 192 mensagens inteiramente sintéticas: 24 por classe, seis
famílias por classe e quatro mensagens por família. Cada classe possui oito
mensagens fáceis, oito médias e oito difíceis, totalizando 64 de cada nível.
Todos os registros usam `split="challenge"` e `dataset_version="challenge-v1"`.

## Famílias novas

- `BUSCA_IMOVEL`: acessibilidade, terreno para construção, atividade comercial,
  mudança temporária, moradia compartilhada e prazo de mudança;
- `AGENDAR_VISITA`: apresentação remota, acompanhante, instruções de acesso,
  múltiplos imóveis, duração e atraso;
- `DUVIDA_IMOVEL`: luz natural, ruído, internet, utilidades, reformas e regras
  do condomínio;
- `DOCUMENTACAO_LOCACAO`: autônomo, estudante, locação empresarial,
  corresponsáveis, correção cadastral e renovação;
- `FINANCEIRO_LOCACAO`: pagamento dividido, canal de pagamento, recibo,
  devolução de garantia, multa de saída e saldo credor;
- `MANUTENCAO_IMOVEL`: pragas, interfone, aquecimento de água, elevador, área
  comum e inspeção preventiva;
- `ATENDIMENTO_HUMANO`: acessibilidade, idioma, mediação, retorno posterior,
  privacidade e múltiplos assuntos;
- `FORA_DE_ESCOPO`: estudo, saúde, compras, viagem, orçamento pessoal e defeito
  em aparelho particular.

Cada registro contém no campo `notes` uma descrição curta do objetivo da
família. Nenhum nome de família anterior foi reutilizado.

## Independência e similaridade

O validador compara IDs, famílias, textos exatos e textos normalizados contra
treino, validação e teste. Uma revisão adicional usa `SequenceMatcher` local,
com limiar 0,88, tanto internamente quanto entre os quatro conjuntos. O conjunto
congelado não possui correspondências nesse limiar. A verificação é textual e
não executa embeddings ou modelos; portanto, não prova independência semântica
absoluta.

## Integridade

`challenge_manifest.json` registra versão, arquivo, SHA-256, distribuições,
classes, schema e declaração de não utilização. O comando abaixo valida o
contrato, recalcula o hash e compara o manifesto:

```bash
python -m imob_ai.intent.challenge
```

O manifesto não contém data ou horário variável e é gerado deterministicamente.

## Segurança e limitações

Não foram usadas mensagens reais, dados de clientes, endereços completos,
telefones, e-mails, documentos ou ocorrências reais. O conjunto é pequeno,
balanceado artificialmente e totalmente sintético. Ele não representa toda a
linguagem real e não deve sustentar sozinho uma conclusão de produção.
