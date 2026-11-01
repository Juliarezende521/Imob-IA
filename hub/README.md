# Local Hugging Face preparation

This directory contains versioned templates and a publication checklist. Generated
packages are written to the ignored `build/huggingface/` directory.

```bash
python -m pip install -e '.[hub,dev]'
python -m imob_ai.hub.prepare
python -m imob_ai.hub.prepare --validate-only
```

These commands never authenticate, upload, create a remote repository, or access
the Hugging Face API. The package status is `READY_FOR_REVIEW`; publication is a
separate manual action. Source code is MIT, the synthetic dataset is CC BY 4.0,
and the model artifact is Apache 2.0.
