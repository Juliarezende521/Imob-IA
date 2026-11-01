---
language:
  - pt
pipeline_tag: text-classification
library_name: skops
tags:
  - scikit-learn
  - tf-idf
  - logistic-regression
license: apache-2.0
---

# ImobAI Portuguese Intent Baseline

This educational baseline classifies short synthetic Portuguese real-estate
messages into eight intents. It combines word TF-IDF, character `char_wb` TF-IDF,
and Logistic Regression.

**Version:** $version  
**Model SHA-256:** `$model_hash`  
**Dataset content SHA-256:** `$dataset_hash`  
**Source project:** $github_url  
**License:** Apache License 2.0 (`apache-2.0`)

**License status:** `$license_status`

Apache 2.0 applies to this model artifact and its package files. It does not
replace or alter the licenses of scikit-learn, NumPy, skops, or other libraries.

## Architecture and training

- word TF-IDF: unigrams and bigrams, Unicode accent stripping, sublinear TF;
- character TF-IDF: `char_wb`, 3–5 characters, `min_df=2`;
- `FeatureUnion` of both representations;
- Logistic Regression: `C=1.0`, `solver="lbfgs"`, `max_iter=2000`, seed 42;
- final package fit: 256 train + 64 validation records (320 total);
- test and final challenge records are excluded from fitting.

The configuration was selected earlier using train and validation only. This
package reproduces the baseline initialized by the local API; it does not perform
new model selection.

## Labels

$labels

## Frozen evaluation results

| Evaluation | Accuracy | Macro F1 | Correct |
| --- | ---: | ---: | ---: |
| Validation during selection | 0.766 | 0.764 | — |
| Previously observed test | 0.781 | 0.786 | 50/64 |
| Final challenge, opened once | 0.682 | 0.673 | 131/192 |

These are previously frozen project results. The challenge was not executed
again, is not included in either package, and must not be published. The dataset
is small and synthetic, so these values do not imply production readiness or
statistical generalization.

The experimental multilingual Transformer performed worse than this baseline on
the observed test and challenge evaluations. Its weights are not included. In
this project, higher complexity did not produce better observed quality.

## Local secure loading

Only load a reviewed local artifact whose SHA-256 matches the manifest. Do not
automatically trust types reported by an unknown file.

```python
from pathlib import Path
from skops.io import get_untrusted_types, load

path = Path("model.skops")
untrusted = get_untrusted_types(file=path)
if untrusted:
    raise RuntimeError(f"Unreviewed types: {untrusted}")
model = load(path, trusted=[])
prediction = model.predict(["Quero alugar uma casa em Franca"])[0]
```

## Input and output

Input is a non-empty Portuguese string. Output is one label. `predict_proba`
produces `model_score`, a relative model value that is **not a calibrated
probability or certainty**. The local application applies a separately documented
operational policy and routes uncertain cases to human review.

See `example_input.json` and `example_output.json` for synthetic examples.

## Intended use

- local educational demonstration;
- reproducible comparison baseline;
- study of traditional NLP for small synthetic datasets.

## Out-of-scope use, limitations, and safety

Do not use this model for legal, financial, contractual, housing eligibility,
tenant screening, emergency, or autonomous customer-service decisions. It was
not trained on real conversations and may fail on dialects, spelling variation,
ambiguous requests, code-switching, or unseen scenarios. Human escalation is
required for uncertain or consequential cases.

No real messages or personal data are included. This model is not ready for
production. Before any public or production use, review privacy, representative
data, fairness, monitoring, security, dependency licenses, and incident handling.
