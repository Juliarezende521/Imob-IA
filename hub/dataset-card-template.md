---
language:
  - pt
task_categories:
  - text-classification
pretty_name: ImobAI Synthetic Portuguese Real Estate Intents
size_categories:
  - n<1K
license: cc-by-4.0
---

# ImobAI Synthetic Portuguese Real Estate Intents

> Short description in Portuguese: conjunto educacional e inteiramente sintético
> para classificação de intenções imobiliárias.

## Dataset description

This dataset contains 384 synthetic Portuguese messages for educational research
on single-label text classification. It includes eight balanced intent classes
and no messages collected from people, companies, or real customer interactions.

**Version:** $version  
**Content SHA-256:** `$content_hash`  
**Source project:** $github_url  
**License:** Creative Commons Attribution 4.0 International (`cc-by-4.0`)

**License status:** `$license_status`

The dataset license applies to the synthetic dataset package. Project source
code and the model artifact have separate licenses.

## Structure and splits

| Split | Records |
| --- | ---: |
| train | 256 |
| validation | 64 |
| test | 64 |

Each of the eight labels has 48 records in total. Scenario families are isolated
by split to reduce template leakage. Exact and normalized duplicate checks are
part of package validation.

Fields:

- `id`: stable synthetic identifier;
- `text`: synthetic Portuguese message;
- `label`: intent label;
- `split`: train, validation, or test;
- `scenario_family`: synthetic scenario family used for split isolation;
- `difficulty`: easy, medium, or hard.

## Labels

$labels

## Creation and privacy

Messages were created programmatically and manually as controlled synthetic
examples. The export is checked for email addresses, phone numbers, Brazilian
document patterns, complete street addresses, private URLs, secrets, and
duplicates. Internal notes are deliberately excluded.

The separate final challenge set is **not included** and must not be published
with this package.

## Loading locally

```python
import json
from pathlib import Path

records = [
    json.loads(line)
    for line in Path("data/train.jsonl").read_text(encoding="utf-8").splitlines()
]
```

## Intended use

- educational experiments in Portuguese intent classification;
- reproducibility demonstrations;
- comparison of simple text-classification baselines.

## Limitations, biases, and responsible use

The dataset is small, synthetic, domain-specific, and does not represent the
full variation of Brazilian Portuguese or real housing interactions. Scenario
choices and writing style may encode authoring bias. It must not be used for
tenant screening, credit, legal, financial, contractual, safety-critical, or
fully automated customer-service decisions.

Observed project results do not demonstrate production readiness. Real-world use
would require consented representative data, privacy review, fairness analysis,
monitoring, human escalation, and a new evaluation protocol.
