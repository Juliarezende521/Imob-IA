# Catálogo sintético de imóveis

O arquivo `properties.json` contém 48 imóveis inteiramente fictícios. Os
registros foram escritos para cobrir combinações variadas de finalidade, tipo,
cidade, preço, área, comodidades e disponibilidade. As descrições usam
vocabulário variado para permitir futuros experimentos de busca literal e
semântica.

Os registros não representam imóveis, ofertas, endereços ou pessoas reais. O
catálogo não deve ser utilizado para decisões comerciais.

## Campos

| Campo | Tipo | Regra |
| --- | --- | --- |
| `id` | texto | identificador único e não vazio |
| `purpose` | enum | `aluguel` ou `venda` |
| `property_type` | enum | `casa`, `apartamento`, `comercial` ou `terreno` |
| `city` | texto | cidade usada somente no cenário fictício |
| `neighborhood` | texto | bairro genérico ou fictício, nunca endereço completo |
| `price` | número | valor positivo em reais |
| `bedrooms` | inteiro | zero ou maior |
| `bathrooms` | inteiro | zero ou maior |
| `parking_spaces` | inteiro | zero ou maior |
| `area_m2` | número | área positiva em metros quadrados |
| `pets_allowed` | booleano | indica aceitação de animais |
| `furnished` | booleano | indica se está mobiliado |
| `features` | lista | características normalizadas pelo schema |
| `description` | texto | descrição não vazia e fictícia |
| `available` | booleano | disponibilidade no cenário simulado |

## Carregamento e validação

```python
from imob_ai.data import load_catalog

properties = load_catalog("data/synthetic/properties.json")
```

O loader detecta arquivo inexistente, JSON malformado, estrutura diferente de
uma lista, catálogo vazio, registros incompatíveis com o schema e IDs
duplicados. A validação é executada por Pydantic.

## Limitações

- a amostra é pequena e não representa o mercado imobiliário;
- cidades e faixas de preço não seguem estatísticas reais;
- a diversidade textual foi criada manualmente e pode conter vieses de escrita;
- não há localização exata, histórico temporal ou informação geográfica;
- o catálogo ainda não possui rótulos de relevância para avaliar buscas.
