# API local

## Escopo

A API FastAPI oferece uma interface local e tipada para componentes já
avaliados no projeto. Ela usa somente dados sintéticos, não chama serviços
externos e não gera respostas com LLM. A busca lexical é o recuperador padrão;
o classificador padrão é o baseline TF-IDF de palavras e caracteres com
Regressão Logística.

O Transformer experimental não é importado, carregado nem necessário para
executar a API. Pesos, embeddings e caches também não fazem parte do serviço.

## Arquitetura e ciclo de vida

No `lifespan` da aplicação, o catálogo é validado e o classificador é ajustado
uma única vez com os 320 exemplos dos splits `train` e `validation`. Os splits
`test` e `challenge` nunca participam dessa inicialização. As dependências
prontas são guardadas em `app.state` e reutilizadas em todas as requisições.

O fluxo é:

```text
requisição JSON
→ validação Pydantic
→ dependência compartilhada
→ classificador ou busca lexical
→ política determinística de roteamento
→ resposta JSON tipada
```

Falha de inicialização mantém `/health/live` disponível, mas faz
`/health/ready` e os endpoints dependentes responderem `503`. Erros de entrada
respondem `422`; falhas internas respondem `500` sem expor stack trace ou
caminhos privados.

## Execução

```bash
python -m pip install -e '.[dev]'
uvicorn imob_ai.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI: <http://127.0.0.1:8000/openapi.json>

O bind em `127.0.0.1` evita exposição involuntária na rede local.

## Endpoints

| Método | Caminho | Responsabilidade |
| --- | --- | --- |
| GET | `/health/live` | Confirma que o processo responde. |
| GET | `/health/ready` | Confirma catálogo, classificador e configuração. |
| GET | `/v1/meta` | Expõe versão e capacidades sem dados privados. |
| POST | `/v1/intents/classify` | Classifica uma mensagem curta em uma das oito intenções. |
| POST | `/v1/properties/search` | Aplica filtros e ranking lexical ao catálogo sintético. |
| POST | `/v1/assistant/route` | Classifica e escolhe uma ação segura e determinística. |

### Classificação

```json
{"message": "Quero visitar o apartamento amanhã"}
```

A resposta contém intenção principal e secundária, scores do modelo, margem,
indicação de revisão humana e nome do modelo. Os scores são valores usados pelo
classificador; não devem ser interpretados como certeza objetiva.

### Busca

```json
{"text": "casa com quintal", "city": "Franca", "purpose": "aluguel", "limit": 3}
```

Filtros incluem finalidade, tipo, cidade, bairro, faixas de preço e área,
quantidades mínimas, animais, mobília e disponibilidade. A resposta identifica
explicitamente o motor `lexical` e que os dados são sintéticos.

### Roteamento

```json
{"message": "Quero falar com um atendente"}
```

Somente `BUSCA_IMOVEL` aciona a busca. Demais intenções produzem ações seguras,
como orientação genérica, identificação do imóvel ou encaminhamento humano.
Classificações incertas sempre viram `human_review`. Não existe geração livre.

## Política de incerteza

A política foi calibrada somente nos 64 exemplos sintéticos de validação. Foram
avaliadas as combinações cartesianas de scores mínimos `{0,00; 0,20; 0,25;
0,30; 0,35}` e margens mínimas `{0,00; 0,02; 0,05; 0,10; 0,15}`, exigindo
cobertura mínima de 50%. O critério foi maximizar a acurácia seletiva, depois a
cobertura e, por fim, preferir limites mais conservadores.

A configuração escolhida foi score mínimo `0,25` e margem mínima `0,02`:
41/64 exemplos cobertos (`0,640625`), 38/41 corretos entre os cobertos
(`0,926829`) e 23 encaminhados para revisão. O registro completo e reproduzível
está em `docs/experiments/api-uncertainty-policy.json`.

Essa política é uma decisão operacional limitada por uma amostra sintética
pequena. Ela não mede confiança universal e deve ser recalibrada antes de
qualquer outro contexto.

## Segurança e privacidade

- corpos das mensagens não são registrados nos logs;
- tamanho máximo da mensagem: 500 caracteres;
- campos desconhecidos são rejeitados;
- CORS não é habilitado;
- não há rede externa no fluxo da aplicação;
- não há dados pessoais, endereços completos ou credenciais;
- catálogo e mensagens são inteiramente fictícios.

Para produção ainda seriam indispensáveis autenticação, autorização, TLS,
rate limiting, observabilidade com política de retenção, gestão de segredos,
revisão de privacidade, monitoramento de drift e testes de carga.

## Decisões

O baseline tradicional foi escolhido por ter vencido a comparação congelada,
ser pequeno, rápido e reproduzível. A busca lexical continua padrão porque a
busca híbrida permanece experimental. O Transformer e um LLM adicionariam
custo, artefatos e riscos operacionais sem benefício demonstrado para este
serviço local.
