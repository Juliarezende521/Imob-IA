# Dataset sintético de intenções

`synthetic_messages.jsonl` contém 384 mensagens curtas em português, todas
escritas manualmente para este experimento. Não há mensagens de clientes,
endereços completos, telefones, e-mails, documentos pessoais ou dados reais.

Cada linha JSON possui:

- `id`: identificador único;
- `text`: mensagem sintética;
- `intent`: uma das oito intenções controladas;
- `split`: `train`, `validation` ou `test`;
- `scenario_family`: situação comunicativa usada para impedir vazamento;
- `difficulty`: `easy`, `medium` ou `hard`;
- `notes`: declaração de origem sintética.

São 48 exemplos por intenção: 32 em treino, 8 em validação e 8 em teste.
Cada família pertence integralmente a um único split. Assim, variações da mesma
situção não aparecem dos dois lados de uma avaliação. O loader rejeita IDs e
textos repetidos, duplicatas após normalização, famílias cruzando splits,
classes insuficientes, ausência de classes e padrões básicos de dados pessoais.

Para validar e resumir:

```bash
python -m imob_ai.intent.dataset
```

Limitações: o conjunto é pequeno, balanceado artificialmente e não cobre toda a
linguagem, ambiguidade, ruído ou distribuição de uso real. Ele serve para
aprendizado e comparação controlada; não deve orientar atendimento ou decisões
comerciais reais.
