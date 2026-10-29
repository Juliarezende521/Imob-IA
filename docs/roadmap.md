# Roadmap técnico

O avanço entre fases depende de revisão e de evidências registradas.

## Fase 0 — Fundação (atual)

- estrutura do repositório;
- pacote Python instalável;
- teste mínimo;
- política de dados e documentação inicial.

## Fase 1 — Catálogo sintético

- definir o schema dos imóveis;
- criar e documentar um catálogo totalmente fictício;
- validar tipos, campos obrigatórios e regras do domínio;
- testar o carregamento dos dados.

## Fase 2 — Busca estruturada

- implementar filtros explícitos como baseline;
- testar resultados, combinações de filtros e casos vazios;
- registrar comportamento e limitações.

## Fase 3 — Busca semântica

- implementar embeddings e busca semântica;
- comparar os resultados com a busca estruturada;
- avaliar Precision@K, Recall@K, MRR e latência.

## Fase 4 — Classificação de mensagens

- definir taxonomia e critérios das classes;
- gerar e validar mensagens sintéticas;
- separar treino, validação e teste;
- comparar modelos e registrar precision, recall, F1 e matriz de confusão;
- analisar erros e versionar o artefato candidato.

## Fase 5 — RAG, produto e publicação

- adicionar geração apenas após validar a recuperação;
- citar fontes, recusar sem evidência e medir respostas sem suporte;
- disponibilizar API FastAPI e demonstração;
- preparar Model Card, Hugging Face, artigo e publicação manual.

