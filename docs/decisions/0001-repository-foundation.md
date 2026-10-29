# ADR 0001: fundação do repositório

- Status: aceita
- Data: 2026-10-29

## Contexto

O projeto precisa começar pequeno, mas deve permitir que exploração, treino,
avaliação e inferência evoluam sem concentrar toda a lógica em notebooks.

## Decisão

Usar layout `src`, testes separados, configuração fora do código e diretórios
explícitos para dados e artefatos. Dados gerados e modelos locais não serão
versionados por padrão.

## Alternativa considerada

Um único notebook permitiria um início rápido, porém dificultaria testes,
reutilização na futura API e separação entre treino e inferência.

## Consequências

Há um pequeno custo inicial de organização. Em troca, o código poderá ser
importado, testado e reutilizado por notebooks, pipelines e API.

