"""
Query Queue for Schemify exploration.

Replaces the iteration-based model with a unified queue of discovery
and completion queries. Each query represents a focused search operation.
"""

import itertools
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import logging

from .models import RecordSet, SchemaAttribute

logger = logging.getLogger("schemify.query_queue")


class QueryType(Enum):
    """Types of queries in the queue."""
    # Discovery queries - find new entities
    DISCOVERY_BROAD = "discovery_broad"           # Initial broad search
    DISCOVERY_SINGLE = "discovery_single"         # Single attribute constraint
    DISCOVERY_PAIR = "discovery_pair"             # Two attribute constraints
    DISCOVERY_TRIPLE = "discovery_triple"         # Three attribute constraints
    DISCOVERY_REFLECTIVE = "discovery_reflective" # LLM-generated from reflection
    
    # Completion queries - fill missing attributes on existing entities
    COMPLETION = "completion"


class QueryPriority(Enum):
    """Priority levels for query scheduling."""
    HIGH = 1      # Completion queries, broad discovery
    MEDIUM = 2    # Single attribute queries
    LOW = 3       # Pair queries
    LOWEST = 4    # Triple queries


@dataclass
class ExplorationQuery:
    """
    A single query in the exploration queue.
    
    This is the atomic unit of work - one query = one web search + extraction.
    """
    id: str
    query_type: QueryType
    priority: QueryPriority
    
    # Attribute constraints for discovery queries
    attribute_constraints: dict[str, str] = field(default_factory=dict)
    
    # For completion queries
    target_record_label: Optional[str] = None
    target_attributes: list[str] = field(default_factory=list)
    
    # Execution tracking
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    entities_found: int = 0
    values_filled: int = 0
    
    # Custom focus text (optional override)
    custom_focus: Optional[str] = None
    
    @property
    def is_discovery(self) -> bool:
        return self.query_type.value.startswith("discovery")
    
    @property
    def is_completion(self) -> bool:
        return self.query_type == QueryType.COMPLETION
    
    @property
    def constraint_depth(self) -> int:
        """Number of attribute constraints (0 for broad, 1-3 for constrained)."""
        return len(self.attribute_constraints)
    
    def build_focus_text(self, category: str) -> str:
        """Build the focus text for this query."""
        if self.custom_focus:
            return self.custom_focus
        
        if self.query_type == QueryType.DISCOVERY_BROAD:
            return ""
        
        if self.query_type == QueryType.COMPLETION:
            attrs = ", ".join(self.target_attributes) if self.target_attributes else "missing attributes"
            return f'Find detailed information about "{self.target_record_label}" focusing on: {attrs}'
        
        # Build from attribute constraints
        if not self.attribute_constraints:
            return ""
        
        constraints = list(self.attribute_constraints.items())
        
        if len(constraints) == 1:
            attr, val = constraints[0]
            return f'Find {category} where {attr} is specifically "{val}".'
        
        elif len(constraints) == 2:
            lines = [f'Find {category} that combine these characteristics:']
            for attr, val in constraints:
                lines.append(f'- {attr}: "{val}"')
            return "\n".join(lines)
        
        else:  # 3+
            lines = [f'Find {category} that match ALL of these characteristics:']
            for attr, val in constraints:
                lines.append(f'- {attr}: "{val}"')
            return "\n".join(lines)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class QueryQueueStats:
    """Statistics about queue state and progress."""
    total_queries: int = 0
    pending_discovery: int = 0
    pending_completion: int = 0
    executed_discovery: int = 0
    executed_completion: int = 0
    total_entities_found: int = 0
    total_new_entities: int = 0  # Excludes duplicates
    total_values_filled: int = 0
    total_pruned: int = 0  # Queries pruned due to zero-yield
    
    # Breakdown by constraint depth
    by_depth: dict[int, dict] = field(default_factory=dict)
    
    # Poisoned attribute values (consistently zero-yield)
    poisoned_values: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ActivityLogEntry:
    """Structured log entry for activity tracking."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # query_executed, query_pruned, value_poisoned, etc.
    query_id: str = ""
    query_type: str = ""
    constraints: dict[str, str] = field(default_factory=dict)
    entities_found: int = 0
    new_entities: int = 0
    duplicates: int = 0
    pruned_count: int = 0
    message: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "query_id": self.query_id,
            "query_type": self.query_type,
            "constraints": self.constraints,
            "entities_found": self.entities_found,
            "new_entities": self.new_entities,
            "duplicates": self.duplicates,
            "pruned_count": self.pruned_count,
            "message": self.message,
        }
    
    def to_log_line(self) -> str:
        """Format as human-readable log line."""
        ts = self.timestamp.strftime("%H:%M:%S")
        if self.event_type == "query_executed":
            constraint_str = ", ".join(f"{k}={v}" for k, v in self.constraints.items()) if self.constraints else "broad"
            return f"[{ts}] {self.query_id} ({self.query_type}): {constraint_str} → {self.new_entities} new, {self.duplicates} dups"
        elif self.event_type == "query_pruned":
            return f"[{ts}] PRUNED {self.pruned_count} queries: {self.message}"
        elif self.event_type == "value_poisoned":
            return f"[{ts}] POISONED: {self.message}"
        else:
            return f"[{ts}] {self.event_type}: {self.message}"


class QueryQueue:
    """
    Manages the queue of exploration queries.
    
    Generates queries from attribute value combinations and schedules
    them with appropriate priorities. Supports both discovery (finding
    new entities) and completion (filling missing attributes) queries.
    
    Features:
    - Bidirectional pruning: zero-yield queries prune extensions AND siblings
    - Poison tracking: attribute values that consistently yield nothing are marked
    - Duplicate awareness: queries returning only duplicates count as zero-yield
    - Structured activity logging for debugging and analytics
    """
    
    def __init__(self, cardinality_threshold: int = 50, seed: int | None = None):
        self.cardinality_threshold = cardinality_threshold
        
        # Seeded RNG for reproducible query ordering
        self._rng = random.Random(seed)
        
        # The queue (actually a list for priority sorting)
        self._pending: list[ExplorationQuery] = []
        self._executed: list[ExplorationQuery] = []
        
        # Tracking
        self._query_counter = 0
        self._explored_constraints: set[str] = set()
        
        # Closed-set attributes for combinatorial generation
        self._closed_attributes: list[SchemaAttribute] = []
        
        # Poison tracking: attribute values that yield nothing
        # Key: (attr_name, value), Value: consecutive_failures
        self._value_failure_count: dict[tuple[str, str], int] = {}
        self._poisoned_values: set[tuple[str, str]] = set()
        self._poison_threshold: int = 2  # Mark as poisoned after N consecutive failures
        
        # Pruning stats
        self._total_pruned: int = 0
        
        # Structured activity log
        self._activity_log: list[ActivityLogEntry] = []
    
    def reset(self):
        """Reset the queue for a new exploration run."""
        self._pending = []
        self._executed = []
        self._query_counter = 0
        self._explored_constraints = set()
        self._closed_attributes = []
        self._value_failure_count = {}
        self._poisoned_values = set()
        self._total_pruned = 0
        self._activity_log = []
        # Preserve _rng (keeps seed)
    
    def _next_id(self) -> str:
        self._query_counter += 1
        return f"Q{self._query_counter:04d}"
    
    def _constraint_key(self, constraints: dict[str, str]) -> str:
        """Generate unique key for a set of constraints."""
        items = sorted(constraints.items())
        return "|".join(f"{k}={v}" for k, v in items)
    
    def _log_activity(self, entry: ActivityLogEntry):
        """Add an entry to the activity log."""
        self._activity_log.append(entry)
        logger.debug(entry.to_log_line())
    
    def get_activity_log(self) -> list[ActivityLogEntry]:
        """Get the structured activity log."""
        return self._activity_log
    
    def export_activity_log(self, path: str, format: str = "json"):
        """Export activity log to file."""
        import json
        import os
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self._activity_log], f, indent=2)
        else:  # text
            with open(path, "w", encoding="utf-8") as f:
                for entry in self._activity_log:
                    f.write(entry.to_log_line() + "\n")

    def add_broad_discovery(self):
        """Add initial broad discovery query."""
        query = ExplorationQuery(
            id=self._next_id(),
            query_type=QueryType.DISCOVERY_BROAD,
            priority=QueryPriority.HIGH,
        )
        self._pending.append(query)
        logger.info(f"Added broad discovery query: {query.id}")
    
    def add_query(self, query: ExplorationQuery):
        """Add a pre-built query to the queue."""
        # Assign ID if not set
        if not query.id or query.id.startswith("reflect_"):
            query.id = self._next_id()
        self._pending.append(query)
        logger.debug(f"Added query: {query.id} [{query.query_type.value}]")
    
    def add_single_constraint_queries(self, attributes: list[SchemaAttribute]):
        """
        Generate single-attribute constraint queries from closed-set attributes.
        
        For each closed-set attribute, creates one query per exploration value.
        """
        count = 0
        for attr in attributes:
            if not attr.is_closed_set or not attr.exploration_values:
                continue
            
            for value in attr.exploration_values:
                constraints = {attr.name: value}
                key = self._constraint_key(constraints)
                
                if key not in self._explored_constraints:
                    query = ExplorationQuery(
                        id=self._next_id(),
                        query_type=QueryType.DISCOVERY_SINGLE,
                        priority=QueryPriority.MEDIUM,
                        attribute_constraints=constraints,
                    )
                    self._pending.append(query)
                    count += 1
        
        logger.info(f"Added {count} single-constraint queries")
        return count
    
    def add_pair_constraint_queries(
        self, 
        attributes: list[SchemaAttribute],
        max_values_per_attr: int = 10,
        max_total_pairs: int = 200,
    ):
        """
        Generate pair-constraint queries from closed-set attributes.
        
        Limits cardinality to avoid explosion:
        - Only uses attributes with <= 20 provisional values
        - Takes top N values per attribute
        - Caps total pairs generated
        """
        suitable_attrs = [
            a for a in attributes 
            if a.is_closed_set and a.exploration_values and len(a.exploration_values) <= 20
        ]
        
        if len(suitable_attrs) < 2:
            logger.info("Not enough suitable attributes for pair queries")
            return 0
        
        pairs = []
        for attr1, attr2 in itertools.combinations(suitable_attrs, 2):
            values1 = attr1.exploration_values[:max_values_per_attr]
            values2 = attr2.exploration_values[:max_values_per_attr]
            
            for val1, val2 in itertools.product(values1, values2):
                constraints = {attr1.name: val1, attr2.name: val2}
                key = self._constraint_key(constraints)
                
                if key not in self._explored_constraints:
                    pairs.append(constraints)
        
        # Shuffle and cap
        self._rng.shuffle(pairs)
        pairs = pairs[:max_total_pairs]
        
        for constraints in pairs:
            query = ExplorationQuery(
                id=self._next_id(),
                query_type=QueryType.DISCOVERY_PAIR,
                priority=QueryPriority.LOW,
                attribute_constraints=constraints,
            )
            self._pending.append(query)
        
        logger.info(f"Added {len(pairs)} pair-constraint queries")
        return len(pairs)
    
    def add_triple_constraint_queries(
        self,
        attributes: list[SchemaAttribute],
        max_values_per_attr: int = 5,
        max_total_triples: int = 100,
    ):
        """
        Generate triple-constraint queries (very selective).
        
        Only uses attributes with <= 10 values each.
        """
        suitable_attrs = [
            a for a in attributes
            if a.is_closed_set and a.exploration_values and len(a.exploration_values) <= 10
        ]
        
        if len(suitable_attrs) < 3:
            logger.info("Not enough suitable attributes for triple queries")
            return 0
        
        triples = []
        for attr1, attr2, attr3 in itertools.combinations(suitable_attrs[:5], 3):
            values1 = attr1.exploration_values[:max_values_per_attr]
            values2 = attr2.exploration_values[:max_values_per_attr]
            values3 = attr3.exploration_values[:max_values_per_attr]
            
            for val1, val2, val3 in itertools.product(values1, values2, values3):
                constraints = {
                    attr1.name: val1,
                    attr2.name: val2,
                    attr3.name: val3,
                }
                key = self._constraint_key(constraints)
                
                if key not in self._explored_constraints:
                    triples.append(constraints)
        
        self._rng.shuffle(triples)
        triples = triples[:max_total_triples]
        
        for constraints in triples:
            query = ExplorationQuery(
                id=self._next_id(),
                query_type=QueryType.DISCOVERY_TRIPLE,
                priority=QueryPriority.LOWEST,
                attribute_constraints=constraints,
            )
            self._pending.append(query)
        
        logger.info(f"Added {len(triples)} triple-constraint queries")
        return len(triples)
    
    def add_completion_query(
        self,
        record_label: str,
        missing_attributes: list[str],
        priority: QueryPriority = QueryPriority.HIGH,
    ):
        """Add a completion query for a specific record."""
        query = ExplorationQuery(
            id=self._next_id(),
            query_type=QueryType.COMPLETION,
            priority=priority,
            target_record_label=record_label,
            target_attributes=missing_attributes,
        )
        self._pending.append(query)
        return query
    
    def add_completion_queries_for_incomplete_records(
        self,
        record_set: RecordSet,
        coverage_threshold: float = 0.6,
        max_queries: int = 20,
    ) -> int:
        """
        Add completion queries for records below coverage threshold.
        
        Returns count of queries added.
        """
        count = 0
        schema_attr_names = {a.name for a in record_set.schema_attributes}
        
        for record in record_set.records:
            if count >= max_queries:
                break
            
            coverage = record.attribute_coverage(record_set.schema_attributes)
            if coverage >= coverage_threshold:
                continue
            
            # Find missing attributes
            missing = [
                name for name in schema_attr_names
                if name not in record.attributes or not record.attributes[name].value
            ]
            
            if missing:
                self.add_completion_query(record.label, missing)
                count += 1
        
        if count > 0:
            logger.info(f"Added {count} completion queries for incomplete records")
        return count
    
    def add_queries_for_new_attribute(
        self,
        new_attr: SchemaAttribute,
        existing_attrs: list[SchemaAttribute],
        include_singles: bool = True,
        include_pairs: bool = True,
        include_triples: bool = False,  # Default off - can explode quickly
    ) -> int:
        """
        Add queries for a newly discovered attribute.
        
        Called during incremental schema evolution when a new attribute
        is promoted to the schema mid-run.
        
        Args:
            new_attr: The newly promoted attribute
            existing_attrs: Existing closed-set attributes for combination queries
            include_singles: Add single-constraint queries for new attr values
            include_pairs: Add pair queries combining new attr with existing
            include_triples: Add triple queries (default off to limit explosion)
            
        Returns:
            Number of queries added
        """
        if not new_attr.is_closed_set or not new_attr.provisional_values:
            logger.info(f"Skipping queries for {new_attr.name}: not closed-set or no values")
            return 0
        
        count = 0
        
        # Track this attribute
        if new_attr not in self._closed_attributes:
            self._closed_attributes.append(new_attr)
        
        # Add single-constraint queries
        if include_singles:
            for value in new_attr.provisional_values:
                constraints = {new_attr.name: value}
                key = self._constraint_key(constraints)
                
                if key not in self._explored_constraints:
                    query = ExplorationQuery(
                        id=self._next_id(),
                        query_type=QueryType.DISCOVERY_SINGLE,
                        priority=QueryPriority.MEDIUM,
                        attribute_constraints=constraints,
                    )
                    self._pending.append(query)
                    count += 1
        
        # Add pair queries with existing attributes
        if include_pairs:
            suitable_existing = [
                a for a in existing_attrs 
                if a.is_closed_set and a.provisional_values and a.name != new_attr.name
            ]
            
            for existing_attr in suitable_existing:
                # Limit values to avoid explosion
                new_values = new_attr.provisional_values[:10]
                existing_values = existing_attr.provisional_values[:10]
                
                for new_val in new_values:
                    for exist_val in existing_values:
                        constraints = {
                            new_attr.name: new_val,
                            existing_attr.name: exist_val,
                        }
                        key = self._constraint_key(constraints)
                        
                        if key not in self._explored_constraints:
                            query = ExplorationQuery(
                                id=self._next_id(),
                                query_type=QueryType.DISCOVERY_PAIR,
                                priority=QueryPriority.LOW,
                                attribute_constraints=constraints,
                            )
                            self._pending.append(query)
                            count += 1
        
        if count > 0:
            self._sort_pending()
            logger.info(f"Added {count} queries for new attribute: {new_attr.name}")
        
        return count

    def populate_from_attributes(
        self,
        attributes: list[SchemaAttribute],
        include_broad: bool = True,
        include_singles: bool = True,
        include_pairs: bool = True,
        include_triples: bool = True,
        lazy: bool = False,
    ):
        """
        Populate the queue with queries generated from attribute values.
        
        This is the main entry point for building the exploration queue.
        
        Args:
            lazy: If True, skip pairs/triples upfront (generate lazily from productive results).
        """
        self._closed_attributes = [a for a in attributes if a.is_closed_set]
        
        if lazy:
            include_pairs = False
            include_triples = False
        
        if include_broad:
            self.add_broad_discovery()
        
        if include_singles:
            self.add_single_constraint_queries(attributes)
        
        if include_pairs:
            self.add_pair_constraint_queries(attributes)
        
        if include_triples:
            self.add_triple_constraint_queries(attributes)
        
        # Sort by priority
        self._sort_pending()
        
        logger.info(f"Queue populated with {len(self._pending)} total queries")
    
    def _sort_pending(self):
        """Sort pending queries by priority."""
        self._pending.sort(key=lambda q: (q.priority.value, q.created_at))
    
    def get_pending_by_depth(self, depth: int) -> list[ExplorationQuery]:
        """
        Get all pending discovery queries at a specific constraint depth.
        
        Args:
            depth: Constraint depth (0=broad, 1=single, 2=pair, 3=triple)
            
        Returns:
            List of pending queries at that depth (excluding poisoned)
        """
        return [
            q for q in self._pending
            if q.is_discovery 
            and q.constraint_depth == depth
            and not self._contains_poisoned_value(q)
        ]
    
    def get_min_pending_depth(self) -> int | None:
        """
        Get the minimum constraint depth with pending queries.
        
        Returns None if no pending discovery queries.
        """
        for depth in range(4):  # 0, 1, 2, 3
            if self.get_pending_by_depth(depth):
                return depth
        return None
    
    def take_queries_at_depth(self, depth: int, max_count: int | None = None) -> list[ExplorationQuery]:
        """
        Remove and return queries at a specific depth.
        
        Args:
            depth: Constraint depth to get queries for
            max_count: Maximum number to take (None = all)
            
        Returns:
            List of queries removed from pending queue
        """
        queries = self.get_pending_by_depth(depth)
        if max_count is not None:
            queries = queries[:max_count]
        
        # Remove from pending
        for q in queries:
            if q in self._pending:
                self._pending.remove(q)
        
        return queries

    def get_next_query(self) -> Optional[ExplorationQuery]:
        """
        Get the next query to execute.
        
        Skips queries that contain poisoned attribute values.
        Returns None if queue is empty.
        """
        while self._pending:
            query = self._pending[0]
            
            # Skip queries containing poisoned values
            if self._contains_poisoned_value(query):
                self._pending.pop(0)
                self._total_pruned += 1
                self._log_activity(ActivityLogEntry(
                    event_type="query_skipped",
                    query_id=query.id,
                    query_type=query.query_type.value,
                    constraints=query.attribute_constraints,
                    message="Contains poisoned attribute value",
                ))
                continue
            
            return query
        
        return None
    
    def _contains_poisoned_value(self, query: ExplorationQuery) -> bool:
        """Check if a query contains any poisoned attribute values."""
        if not query.attribute_constraints:
            return False
        
        for attr, value in query.attribute_constraints.items():
            if (attr, value) in self._poisoned_values:
                return True
        return False
    
    def mark_executed(
        self,
        query: ExplorationQuery,
        entities_found: int = 0,
        new_entities: int = 0,
        values_filled: int = 0,
    ):
        """
        Mark a query as executed and move to executed list.
        
        Enhanced pruning logic:
        1. If zero NEW entities (including all-duplicates), prune extensions
        2. Track failure counts per attribute value
        3. Poison attribute values that consistently fail
        
        Args:
            query: The executed query
            entities_found: Total entities returned (including duplicates)
            new_entities: Only the non-duplicate entities
            values_filled: Attribute values filled (for completion queries)
        """
        query.executed_at = datetime.now()
        query.entities_found = entities_found
        query.values_filled = values_filled
        
        # Use new_entities for yield calculations (duplicates = zero yield)
        effective_yield = new_entities if new_entities is not None else entities_found
        duplicates = entities_found - effective_yield
        
        # Mark constraints as explored
        if query.attribute_constraints:
            key = self._constraint_key(query.attribute_constraints)
            self._explored_constraints.add(key)
        
        # Move from pending to executed
        if query in self._pending:
            self._pending.remove(query)
        self._executed.append(query)
        
        # Log the execution
        self._log_activity(ActivityLogEntry(
            event_type="query_executed",
            query_id=query.id,
            query_type=query.query_type.value,
            constraints=query.attribute_constraints,
            entities_found=entities_found,
            new_entities=effective_yield,
            duplicates=duplicates,
        ))
        
        # Handle zero-yield (no new entities)
        if effective_yield == 0 and query.attribute_constraints:
            # 1. Prune extending queries
            pruned = self._prune_extending_queries(query.attribute_constraints)
            if pruned > 0:
                self._total_pruned += pruned
                self._log_activity(ActivityLogEntry(
                    event_type="query_pruned",
                    query_id=query.id,
                    pruned_count=pruned,
                    message=f"Extensions of {query.attribute_constraints}",
                ))
            
            # 2. Track failure counts for single-value constraints
            if len(query.attribute_constraints) == 1:
                attr, value = list(query.attribute_constraints.items())[0]
                self._track_value_failure(attr, value)
        else:
            # Reset failure counts for successful constraint values
            if query.attribute_constraints:
                for attr, value in query.attribute_constraints.items():
                    key = (attr, value)
                    if key in self._value_failure_count:
                        self._value_failure_count[key] = 0
        
        logger.debug(
            f"Query {query.id} executed: "
            f"{entities_found} found, {effective_yield} new, {duplicates} dups"
        )
    
    def _track_value_failure(self, attr: str, value: str):
        """Track a failure for an attribute value, potentially poisoning it."""
        key = (attr, value)
        self._value_failure_count[key] = self._value_failure_count.get(key, 0) + 1
        
        if self._value_failure_count[key] >= self._poison_threshold:
            if key not in self._poisoned_values:
                self._poisoned_values.add(key)
                
                # Prune all pending queries containing this poisoned value
                pruned = self._prune_queries_with_value(attr, value)
                
                self._log_activity(ActivityLogEntry(
                    event_type="value_poisoned",
                    message=f"{attr}={value} (pruned {pruned} queries)",
                    pruned_count=pruned,
                ))
                
                logger.info(f"Poisoned {attr}={value}: pruned {pruned} queries")
    
    def _prune_queries_with_value(self, attr: str, value: str) -> int:
        """Remove all pending queries that contain a specific attribute value."""
        to_prune = []
        for query in self._pending:
            if query.attribute_constraints and query.attribute_constraints.get(attr) == value:
                to_prune.append(query)
        
        for query in to_prune:
            self._pending.remove(query)
            self._total_pruned += 1
        
        return len(to_prune)
    
    def _prune_extending_queries(self, failed_constraints: dict[str, str]) -> int:
        """
        Remove queries that extend the failed constraint set.
        
        A query "extends" the failed constraints if it contains ALL of them.
        E.g., if {A=1} failed, prune {A=1, B=2} and {A=1, B=2, C=3}.
        
        Returns count of pruned queries.
        """
        if not failed_constraints:
            return 0
        
        to_prune = []
        for query in self._pending:
            if not query.attribute_constraints:
                continue
            
            # Check if query contains all failed constraints
            extends_failed = all(
                query.attribute_constraints.get(attr) == value
                for attr, value in failed_constraints.items()
            )
            
            if extends_failed and len(query.attribute_constraints) > len(failed_constraints):
                to_prune.append(query)
        
        for query in to_prune:
            self._pending.remove(query)
            logger.debug(f"Pruned query {query.id}: {query.attribute_constraints}")
        
        return len(to_prune)
    
    def skip_query(self, query: ExplorationQuery):
        """Skip a query without executing (e.g., if deemed low-yield)."""
        if query in self._pending:
            self._pending.remove(query)
        # Don't add to executed - just discard
    
    def has_pending(self) -> bool:
        """Check if there are pending queries."""
        return len(self._pending) > 0
    
    def pending_count(self) -> int:
        """Count of pending queries."""
        return len(self._pending)
    
    def executed_count(self) -> int:
        """Count of executed queries."""
        return len(self._executed)
    
    def get_stats(self) -> QueryQueueStats:
        """Get detailed statistics about the queue."""
        stats = QueryQueueStats()
        
        stats.total_queries = len(self._pending) + len(self._executed)
        
        stats.pending_discovery = sum(
            1 for q in self._pending if q.is_discovery
        )
        stats.pending_completion = sum(
            1 for q in self._pending if q.is_completion
        )
        stats.executed_discovery = sum(
            1 for q in self._executed if q.is_discovery
        )
        stats.executed_completion = sum(
            1 for q in self._executed if q.is_completion
        )
        
        stats.total_entities_found = sum(q.entities_found for q in self._executed)
        stats.total_values_filled = sum(q.values_filled for q in self._executed)
        stats.total_pruned = self._total_pruned
        stats.poisoned_values = list(self._poisoned_values)
        
        # Breakdown by constraint depth
        for depth in range(4):  # 0, 1, 2, 3
            pending = sum(1 for q in self._pending if q.constraint_depth == depth)
            executed = sum(1 for q in self._executed if q.constraint_depth == depth)
            entities = sum(
                q.entities_found for q in self._executed 
                if q.constraint_depth == depth
            )
            stats.by_depth[depth] = {
                "pending": pending,
                "executed": executed,
                "entities_found": entities,
            }
        
        return stats
    
    def get_pending_by_type(self, query_type: QueryType) -> list[ExplorationQuery]:
        """Get pending queries of a specific type."""
        return [q for q in self._pending if q.query_type == query_type]
    
    def prioritize_completion(self):
        """Re-sort to prioritize completion queries."""
        self._sort_pending()
    
    def interleave_completion_queries(
        self,
        record_set: RecordSet,
        every_n_discovery: int = 10,
        coverage_threshold: float = 0.5,
    ):
        """
        Add completion queries interleaved with discovery queries.
        
        This ensures records get filled in as we discover them,
        rather than waiting until the end.
        
        Args:
            record_set: The record set to check for incomplete records
            every_n_discovery: Add completion queries every N discovery queries
            coverage_threshold: Target incomplete records below this coverage
        """
        discovery_count = sum(1 for q in self._pending if q.is_discovery)
        completion_slots = discovery_count // every_n_discovery
        
        if completion_slots > 0:
            self.add_completion_queries_for_incomplete_records(
                record_set,
                coverage_threshold=coverage_threshold,
                max_queries=completion_slots,
            )
    
    # ========================================================================
    # Lazy Query Generation
    # ========================================================================
    
    def get_productive_values(self, depth: int) -> dict[str, set[str]]:
        """
        Get attribute values that produced new entities at a given constraint depth.
        
        Used for lazy query generation: only generate deeper queries from
        productive parent constraints.
        
        Args:
            depth: Constraint depth (1=single, 2=pair)
            
        Returns:
            Dict mapping attribute_name -> set of productive values
        """
        productive: dict[str, set[str]] = {}
        for q in self._executed:
            if q.constraint_depth == depth and q.entities_found > 0:
                for attr, value in q.attribute_constraints.items():
                    productive.setdefault(attr, set()).add(value)
        return productive
    
    def generate_pairs_from_productive(
        self,
        productive_values: dict[str, set[str]],
        attributes: list[SchemaAttribute],
        max_values_per_attr: int = 10,
        max_total_pairs: int = 200,
    ) -> int:
        """
        Generate pair-constraint queries using only productive single-constraint values.
        
        Instead of generating all possible pairs upfront, this creates pairs only from
        attribute values that actually produced results as single constraints.
        
        Args:
            productive_values: Dict of attr_name -> set of productive values
                              (from get_productive_values(depth=1))
            attributes: All schema attributes (for naming/lookup)
            max_values_per_attr: Max values per attribute in combinations
            max_total_pairs: Maximum total pair queries to generate
            
        Returns:
            Number of queries added
        """
        if len(productive_values) < 2:
            logger.info("Not enough productive attributes for lazy pair queries")
            return 0
        
        attr_map = {a.name: a for a in attributes if a.is_closed_set}
        
        # Build (attr_name, values) list from productive results
        productive_attrs = [
            (name, list(values)[:max_values_per_attr])
            for name, values in productive_values.items()
            if name in attr_map
        ]
        
        pairs = []
        for i, (name1, vals1) in enumerate(productive_attrs):
            for name2, vals2 in productive_attrs[i + 1:]:
                for v1 in vals1:
                    for v2 in vals2:
                        constraints = {name1: v1, name2: v2}
                        key = self._constraint_key(constraints)
                        if key not in self._explored_constraints:
                            pairs.append(constraints)
        
        self._rng.shuffle(pairs)
        pairs = pairs[:max_total_pairs]
        
        for constraints in pairs:
            query = ExplorationQuery(
                id=self._next_id(),
                query_type=QueryType.DISCOVERY_PAIR,
                priority=QueryPriority.LOW,
                attribute_constraints=constraints,
            )
            self._pending.append(query)
        
        if pairs:
            self._sort_pending()
        logger.info(f"Lazy generation: added {len(pairs)} pair queries from productive singles")
        return len(pairs)
    
    def generate_triples_from_productive(
        self,
        productive_values: dict[str, set[str]],
        attributes: list[SchemaAttribute],
        max_values_per_attr: int = 5,
        max_total_triples: int = 100,
    ) -> int:
        """
        Generate triple-constraint queries using only productive pair-constraint values.
        
        Args:
            productive_values: Dict of attr_name -> set of productive values
                              (from get_productive_values(depth=2))
            attributes: All schema attributes
            max_values_per_attr: Max values per attribute
            max_total_triples: Maximum total triple queries
            
        Returns:
            Number of queries added
        """
        if len(productive_values) < 3:
            logger.info("Not enough productive attributes for lazy triple queries")
            return 0
        
        attr_map = {a.name: a for a in attributes if a.is_closed_set}
        productive_attrs = [
            (name, list(values)[:max_values_per_attr])
            for name, values in productive_values.items()
            if name in attr_map
        ][:5]  # Limit attribute count
        
        triples = []
        for i, (n1, v1s) in enumerate(productive_attrs):
            for j, (n2, v2s) in enumerate(productive_attrs[i + 1:], i + 1):
                for n3, v3s in productive_attrs[j + 1:]:
                    for val1 in v1s:
                        for val2 in v2s:
                            for val3 in v3s:
                                constraints = {n1: val1, n2: val2, n3: val3}
                                key = self._constraint_key(constraints)
                                if key not in self._explored_constraints:
                                    triples.append(constraints)
        
        self._rng.shuffle(triples)
        triples = triples[:max_total_triples]
        
        for constraints in triples:
            query = ExplorationQuery(
                id=self._next_id(),
                query_type=QueryType.DISCOVERY_TRIPLE,
                priority=QueryPriority.LOWEST,
                attribute_constraints=constraints,
            )
            self._pending.append(query)
        
        if triples:
            self._sort_pending()
        logger.info(f"Lazy generation: added {len(triples)} triple queries from productive pairs")
        return len(triples)
