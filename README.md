# Imob AI

Projeto de **engenharia de IA** aplicado a um problema imobiliário sintético.
A prioridade é construir uma solução tecnicamente sólida, com Machine Learning,
avaliação reproduzível, busca semântica e RAG com fontes.

> Status: recuperação lexical, semântica e híbrida, classificação local de
> intenções e API FastAPI para demonstração. A busca lexical e o classificador
> tradicional permanecem como padrões.

## Objetivo técnico

O projeto evoluirá em incrementos revisáveis:

1. catálogo fictício e busca estruturada de imóveis como baseline;
2. busca semântica avaliada com métricas de recuperação;
3. RAG baseado nas evidências recuperadas;
4. classificador de mensagens com dados exclusivamente sintéticos;
5. API, testes, documentação e publicação manual de artefatos.

Cada incremento deverá incluir justificativa técnica, teste e evidência de
avaliação antes de avançar.

As buscas e avaliações estão documentadas em
[`docs/structured-search.md`](docs/structured-search.md),
[`docs/baseline-evaluation.md`](docs/baseline-evaluation.md) e
[`docs/semantic-search.md`](docs/semantic-search.md). A fusão dos rankings está
em [`docs/hybrid-search.md`](docs/hybrid-search.md).
O classificador está descrito em
[`docs/intent-classification.md`](docs/intent-classification.md).
Existe também um challenge final sintético e congelado, documentado em
[`docs/intent-challenge-set.md`](docs/intent-challenge-set.md). O experimento
Transformer local e sua comparação final estão documentados em
[`docs/intent-transformer.md`](docs/intent-transformer.md) e
[`docs/intent-transformer-evaluation.md`](docs/intent-transformer-evaluation.md).
No conjunto atual, o baseline tradicional continua sendo a opção recomendada.
A API local está documentada em [`docs/api.md`](docs/api.md), com validação e
benchmark em [`docs/api-evaluation.md`](docs/api-evaluation.md).

## API local

Após preparar o ambiente, inicie o servidor apenas na interface local:

```bash
uvicorn imob_ai.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

A documentação interativa fica em <http://127.0.0.1:8000/docs>. A API expõe
saúde, classificação de intenção, busca lexical e roteamento determinístico.
Ela treina uma única instância do baseline TF-IDF + Regressão Logística na
inicialização, exclusivamente com treino + validação. O Transformer não é
carregado pela API.

Exemplo:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/intents/classify \
  -H 'Content-Type: application/json' \
  -d '{"message":"Quero visitar o apartamento amanhã"}'
```

Todos os dados são sintéticos. Esta é uma demonstração local, sem autenticação,
controle de taxa ou garantias para uso em produção.

## Estrutura

```text
configs/       configurações fora do código
data/          dados sintéticos e dados locais não versionados
docs/          decisões, registros e futuro artigo técnico
models/        modelos locais (não versionados)
notebooks/     exploração e comunicação de experimentos
src/imob_ai/   código Python reutilizável
tests/         testes automatizados
```

## Preparação do ambiente

Requer Python 3.9 ou superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Verificação inicial

```bash
python -m unittest discover -s tests -v
pytest
ruff check .
```

## Dados e privacidade

Nenhum dado real de imobiliária ou cliente será utilizado. Os datasets serão
sintéticos, documentados e validados antes do treinamento.

Consulte também [`docs/privacy.md`](docs/privacy.md) para as regras de segurança
e privacidade do projeto.

## Idiomas

Este é o README principal em português. A versão em inglês está em
[`README.en.md`](README.en.md).

