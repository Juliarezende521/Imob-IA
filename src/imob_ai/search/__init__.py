"""Busca estruturada e lexical sobre o catálogo de imóveis."""

from imob_ai.search.schemas import (
    PropertySearchQuery,
    SearchMatch,
    SearchResult,
    SearchSort,
)
from imob_ai.search.service import search_properties

__all__ = [
    "PropertySearchQuery",
    "SearchMatch",
    "SearchResult",
    "SearchSort",
    "search_properties",
]
