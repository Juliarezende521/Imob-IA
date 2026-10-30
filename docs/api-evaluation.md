# Avaliação da API local

## Validação funcional

A API foi exercitada com testes unitários e de integração usando `TestClient`,
sem internet. A cobertura específica inclui:

- saúde, prontidão e falha controlada de inicialização;
- contratos válidos e entradas vazias, longas, inválidas ou com campos extras;
- classificação e política de revisão humana;
- busca por texto e filtros, limites, intervalos, indisponibilidade e ausência;
- todas as ações determinísticas de roteamento;
- metadados e OpenAPI sem caminhos privados;
- ausência de CORS e de conteúdo de mensagens nos logs;
- bloqueio de acesso à rede;
- reutilização da mesma instância do classificador sob chamadas repetidas e
  concorrentes;
- integração real com treino único em `train + validation`, sem carregar o
  Transformer.

## Verificação HTTP real

Um servidor Uvicorn foi iniciado em `127.0.0.1:8000` e consultado com `curl`.
Os cinco fluxos responderam `200`:

| Fluxo | Resultado observado |
| --- | --- |
| `/health/live` | `alive` |
| `/health/ready` | classificador, catálogo e configuração prontos |
| classificação | `AGENDAR_VISITA`, sem revisão humana |
| busca | 3 resultados sintéticos de aluguel em Franca |
| roteamento | `ATENDIMENTO_HUMANO` → `human_support` |

O processo foi encerrado normalmente após a verificação.

## Benchmark local

O benchmark usa o aplicativo real em memória, uma chamada de aquecimento e 30
medições por operação. Ele mede latência de ponta a ponta pelo `TestClient` e o
tempo de inicialização, no ambiente Python 3.9.6, macOS 15.7.4, x86_64.

| Operação | Mediana | p95 |
| --- | ---: | ---: |
| Inicialização | 217,77 ms | — |
| Classificação | 7,41 ms | 10,22 ms |
| Busca lexical | 5,47 ms | 6,73 ms |
| Roteamento | 8,45 ms | 9,66 ms |

Os valores brutos e a metodologia estão em
`docs/experiments/api-benchmark.json`. São medições locais de demonstração, não
um teste de capacidade, concorrência de produção ou SLA.

## Limitações

- datasets pequenos e totalmente sintéticos;
- apenas 64 mensagens na validação da política de incerteza;
- benchmark em uma única máquina e sem tráfego de rede real;
- ausência intencional de autenticação, persistência, rate limiting e filas;
- classificador treinado na inicialização, adequado ao porte atual, não a uma
  implantação com múltiplos workers;
- roteamento determinístico, sem diálogo ou resposta generativa;
- busca lexical como padrão; sem integração automática da busca experimental.
