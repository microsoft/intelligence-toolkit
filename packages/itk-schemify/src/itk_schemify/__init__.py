"""
Schemify - AI-powered entity extraction and schema discovery system.

Uses web search to ground extractions in real, citable sources.
Employs a query queue for systematic attribute-value exploration.
"""

from .models import (
    AttributeValue,
    BudgetExceededError,
    Citation,
    Record,
    RecordSet,
    SchemaAttribute,
    SchemifyConfig,
    SourcedValue,
)
from .schemify import Schemify, IterationMetrics
from .query_queue import QueryQueue, ExplorationQuery, QueryType, QueryPriority
from .strategy_agentic import AgenticStrategy
from .search import SearchProvider, OpenAISearchProvider
from .resolution import (
    find_fuzzy_match,
    find_all_fuzzy_matches,
    cluster_fuzzy_matches,
    normalize_label,
)

__version__ = "2.3.0"
__all__ = [
    "Schemify",
    "SchemifyConfig",
    "IterationMetrics",
    "QueryQueue",
    "ExplorationQuery", 
    "QueryType",
    "QueryPriority",
    "AgenticStrategy",
    "Record",
    "RecordSet",
    "AttributeValue",
    "SourcedValue",
    "Citation",
    "SchemaAttribute",
    "BudgetExceededError",
    # Search provider abstraction
    "SearchProvider",
    "OpenAISearchProvider",
    # Fuzzy matching utilities
    "find_fuzzy_match",
    "find_all_fuzzy_matches", 
    "cluster_fuzzy_matches",
    "normalize_label",
]
