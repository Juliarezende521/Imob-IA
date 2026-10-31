# Demonstração local com Streamlit

## Finalidade

A interface ImobAI apresenta, de forma visual, os componentes avaliados do
projeto: classificação de intenções, política de incerteza, busca lexical e
roteamento seguro. Ela é uma demonstração educacional local, não um produto de
produção.

Todos os imóveis, mensagens de exemplo e resultados são sintéticos. Não use a
aplicação para decisões comerciais reais nem insira dados pessoais.

## Arquitetura

```text
Usuário
  → Streamlit
  → HTTP local
  → FastAPI
  → classificador TF-IDF + Regressão Logística
  → busca lexical ou roteador seguro
  → JSON validado
  → apresentação no Streamlit
```

A interface não importa classificador, catálogo, busca ou avaliação. O módulo
`ui/api_client.py` é a única fronteira com o backend e valida todas as respostas
com Pydantic. A URL aceita apenas hosts locais.

## Pré-requisitos e instalação

- Python 3.9 ou superior compatível com as versões fixadas;
- ambiente virtual do projeto;
- dependências instaladas.

```bash
cd "/Users/julia/Documents/ChatGPT/imob-ai"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Execução

Terminal 1 — API:

```bash
cd "/Users/julia/Documents/ChatGPT/imob-ai"
source .venv/bin/activate
uvicorn imob_ai.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Terminal 2 — interface:

```bash
cd "/Users/julia/Documents/ChatGPT/imob-ai"
source .venv/bin/activate
IMOB_AI_API_URL=http://127.0.0.1:8000 \
  streamlit run src/imob_ai/ui/app.py
```

O comando usa `create_app --factory` porque o backend adota uma factory com
lifespan, não uma instância global chamada `app`.

O navegador normalmente abre <http://localhost:8501>. A variável
`IMOB_AI_API_URL` é opcional e assume `http://127.0.0.1:8000`; por segurança,
somente `localhost`, `127.0.0.1` e `::1` são aceitos.

## Áreas da interface

### Assistente

Envia uma mensagem ao endpoint `/v1/assistant/route`. Mostra intenção,
`model_score`, incerteza, ação, mensagem fixa do backend e imóveis encontrados.
O valor `model_score` representa força relativa da decisão, não probabilidade
calibrada.

Para buscas, a tela também mostra `Entendi sua busca como`, com filtros
reconhecidos, correções controladas e características usadas no ranking. Uma
localização desconhecida aparece como restrição não resolvida; a busca não é
executada até a mensagem ser reformulada. Esse aviso é diferente de uma busca
válida com zero resultados e de uma falha técnica da API.

### Buscar imóveis

Usa `/v1/properties/search` com texto livre, finalidade, tipo, localização,
preço, área, quantidades mínimas, animais, mobília, limite e ordenação. Os
cartões mostram somente campos recebidos da API.

### Entenda o resultado

Explica as oito intenções, a política de revisão humana, as estratégias de
recuperação e os resultados congelados dos modelos. A interface não reexecuta o
challenge.

### Sobre o projeto

Resume arquitetura, tecnologias, decisões, limitações e execução local.

## Exemplos

- `Procuro uma casa para alugar em Franca.`
- `Quero agendar uma visita.`
- `Meu boleto venceu, preciso falar com alguém.`
- `O chuveiro do imóvel parou de funcionar.`
- `Quero falar com um atendente.`
- `Qual é a previsão do tempo?`

As sugestões nunca são executadas automaticamente.

## Testes

```bash
pytest tests/test_ui_api_client.py tests/test_ui_presenters.py tests/test_ui_config.py
pytest tests/test_query_understanding.py
pytest
ruff check .
ruff format --check .
```

Os testes unitários usam transporte HTTP simulado e não exigem API real,
navegador, internet ou modelos externos.

## Tratamento de erros

A interface diferencia API indisponível, timeout, erro HTTP e resposta
incompatível. A mensagem principal é amigável; o tipo técnico fica em uma área
expansível. Não há repetição automática nem polling contínuo. O estado pode ser
verificado novamente pelo botão `Verificar conexão`.

## Problemas comuns

- **API indisponível:** confirme se o Terminal 1 continua aberto e consulte
  `http://127.0.0.1:8000/health/ready`.
- **Porta ocupada:** encerre o processo anterior ou escolha outra porta local e
  ajuste `IMOB_AI_API_URL`.
- **Módulo não encontrado:** ative `.venv` e reinstale com
  `python -m pip install -e '.[dev]'`.
- **Configuração recusada:** a interface bloqueia URLs externas, HTTPS,
  credenciais e parâmetros na URL intencionalmente.

## Segurança, privacidade e limitações

- nenhum histórico é persistido;
- mensagens integrais não são registradas pela camada HTTP da interface;
- nenhuma telemetria ou analytics foi adicionada;
- a telemetria de uso do Streamlit está explicitamente desativada;
- nenhuma API externa ou paga é chamada;
- o Transformer experimental e seus pesos não são carregados;
- não existe LLM, RAG ou geração livre de respostas;
- não há autenticação, TLS, rate limiting ou banco de dados;
- datasets e catálogo são pequenos e inteiramente sintéticos;
- resultados e thresholds não devem ser generalizados para dados reais;
- a compreensão textual cobre apenas aliases e padrões explícitos documentados;
- o estado temporário do Streamlit desaparece quando a sessão é encerrada.

Antes de exposição pública seriam necessários controles de acesso, revisão de
privacidade, proteção contra abuso, observabilidade segura, testes de carga e
uma estratégia de implantação adequada.
