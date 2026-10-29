"""Normalização textual compartilhada pela busca."""

import re
import unicodedata


def normalize_text(value: str) -> str:
    """Remove acentos, uniformiza caixa e reduz espaços excedentes."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_accents.casefold().split())


def tokenize(value: str) -> list[str]:
    """Retorna termos alfanuméricos distintos na ordem da consulta."""
    terms = re.findall(r"[a-z0-9]+", normalize_text(value))
    return list(dict.fromkeys(terms))
