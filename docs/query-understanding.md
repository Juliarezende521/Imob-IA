# Compreensão estruturada de consultas

## Problema corrigido

Classificar uma mensagem como `BUSCA_IMOVEL` não identifica automaticamente
suas restrições. Antes desta camada, a rota enviava a frase inteira para o
ranking lexical; cidade, tipo e finalidade podiam ser tratados apenas como
palavras, permitindo resultados incompatíveis.

A compreensão estruturada agora fica entre a classificação e a busca:

```text
mensagem → intenção → parser determinístico → filtros AND
                                      └─────→ texto restante para ranking
```

## Resultado tipado

O parser retorna consulta original e normalizada, filtros extraídos, texto
restante, restrições reconhecidas e não resolvidas e a indicação
`requires_clarification`. Cada decisão registra o trecho de origem; a correção
conservadora de `acasa` também aparece explicitamente.

## Vocabulário

- **Finalidade:** aluguel, alugar, alugando, locação, locar, comprar, compra,
  venda e à venda.
- **Tipo:** casa(s), apartamento(s), apê, terreno(s), comercial, sala
  comercial, ponto comercial e imóvel comercial.
- **Localização:** cidades e bairros vêm diretamente do catálogo, com caixa e
  acentos normalizados e preferência pela correspondência mais longa.
- **Padrões adicionais:** preço máximo, quartos ou dormitórios mínimos,
  banheiros, garagem, animais e mobília em construções explícitas.

Não existe corretor ortográfico genérico. O único erro tolerado inicialmente é
`acasa`, como alias explícito de `casa`.

## Política estrita

Finalidade, tipo, cidade e bairro reconhecidos viram filtros combinados com
`AND`. Depois desses filtros, apenas as características restantes participam do
ranking lexical. Nenhum filtro é removido se a busca retornar zero.

Localização desconhecida, valores contraditórios ou conflito entre mensagem e
filtro explícito bloqueiam a busca e solicitam esclarecimento. O sistema não
inventa aproximações, não substitui bairros e não mostra imóveis incompatíveis.

Exemplo:

```text
eu alugar um acasa no city petropolis em franca
```

Interpretação:

- finalidade: aluguel;
- tipo: casa, pela correção controlada de `acasa`;
- cidade: Franca;
- bairro: `city petropolis`, não resolvido;
- ação: pedir esclarecimento, sem executar busca.

## Avaliação das regras

O arquivo `data/evaluation/query_understanding_cases.json` contém 40 consultas
inteiramente sintéticas, separado dos datasets de classificação. Os 30 casos de
desenvolvimento e os 10 casos de teste foram definidos antes da execução do
respectivo split.

```bash
python -m imob_ai.query_understanding.evaluate --split development
python -m imob_ai.query_understanding.evaluate --split test
```

Resultado observado:

| Split | Casos exatos | Finalidade | Tipo | Cidade | Bairro | Não resolvidas | Esclarecimento |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Desenvolvimento | 30/30 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| Teste | 10/10 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |

Esses números medem somente regras determinísticas em um conjunto pequeno e
sintético. Não são métricas de ML, não representam linguagem aberta e não devem
ser generalizados.

## Limitações e evolução

O parser cobre vocabulário e padrões explícitos, não toda a língua portuguesa.
Não resolve referências contextuais, negações amplas, comparações complexas,
faixas flexíveis ou erros ortográficos genéricos. Consultas contraditórias são
encaminhadas para esclarecimento.

Um comparativo futuro pode medir essas mesmas saídas estruturadas com LLM ou
agente, usando o conjunto sintético como contrato. Essa alternativa só deve ser
considerada com avaliação, controles de privacidade, custo e determinismo; ela
não é necessária para corrigir o fluxo atual.
