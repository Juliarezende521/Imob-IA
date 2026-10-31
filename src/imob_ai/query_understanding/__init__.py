"""API pública da compreensão determinística de consultas imobiliárias."""

from imob_ai.query_understanding.parser import interpret_property_query
from imob_ai.query_understanding.schemas import QueryInterpretation

__all__ = ["QueryInterpretation", "interpret_property_query"]
