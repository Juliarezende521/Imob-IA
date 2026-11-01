# Artefatos públicos no Hugging Face

O projeto publica dois artefatos independentes e versionados:

- [dataset sintético](https://huggingface.co/datasets/JuliaRezende521/imob-ai-intent-dataset-pt):
  384 mensagens sintéticas em português, sob CC BY 4.0;
- [modelo de intenções](https://huggingface.co/JuliaRezende521/imob-ai-intent-classifier-pt):
  baseline TF-IDF + Regressão Logística serializado com `skops`, sob Apache 2.0.

O GitHub contém código, testes e documentação. O repositório de dataset contém
somente os splits autorizados; o repositório de modelo contém o artefato treinado,
configuração, labels e exemplos. O challenge permanece privado e não integra
nenhum pacote público.

Revisões imutáveis validadas neste marco:

- dataset: `1c1fdd14a4730a7a616a8e133b1f4f1bf6577c3c`;
- modelo: `a2bc23ea970bf1c4186c6798e6359367e85a7d7f`.

## Baixar e validar

Instale o extra local e execute a validação fixada nas revisões públicas já
auditadas:

```bash
python -m pip install -e '.[hub,dev]'
python -m imob_ai.hub.verify_publication
```

O comando usa a API pública oficial, `token=False`, timeout e IDs validados. Ele
confirma acesso anônimo, baixa apenas a lista permitida de arquivos, confere
manifestações SHA-256, schema, splits, classes, famílias, exclusão do challenge,
licenças, exemplos e o modelo. O cache fica em `build/huggingface/public-cache/`,
fora do Git.

Depois da primeira validação, é possível repetir sem rede:

```bash
python -m imob_ai.hub.verify_publication --offline
```

As opções `--dataset-repo`, `--model-repo`, `--dataset-revision` e
`--model-revision` existem para auditorias explícitas. IDs e revisões inseguros
são rejeitados; prefira sempre commits completos e imutáveis.

## Carregamento seguro e previsão

Nunca carregue um arquivo desconhecido antes da inspeção:

```python
from pathlib import Path
from skops.io import get_untrusted_types, load

path = Path("model.skops")
untrusted = get_untrusted_types(file=path)
if untrusted:
    raise RuntimeError(f"Tipos não revisados: {untrusted}")

model = load(path, trusted=[])
intent = model.predict(["Quero alugar uma casa em Franca"])[0]
scores = model.predict_proba(["Quero alugar uma casa em Franca"])[0]
```

Os scores são valores relativos do modelo e não probabilidades calibradas. O
dataset é inteiramente sintético, não contém mensagens reais ou dados pessoais,
e o modelo não está pronto para produção nem para decisões imobiliárias reais.
