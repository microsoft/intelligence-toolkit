"""
Main Schemify class - the primary interface for the system.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
import logging
import pandas as pd
from IPython.display import display, HTML

from .models import (
    Record, RecordSet, SchemaAttribute, SchemifyConfig, AttributeValue,
    BudgetExceededError
)
from .llm import LLMClient
from .cache import Cache, NoOpCache
from .extraction import ExtractionEngine
from .resolution import ResolutionEngine
from .search import SearchProvider, OpenAISearchProvider
from .query_queue import QueryQueue, ExplorationQuery, QueryType, QueryPriority
from . import prompts

logger = logging.getLogger("schemify")


@dataclass
class IterationMetrics:
    """Metrics from a single discovery query."""
    query_num: int
    query_type: str
    new_entities: int
    new_attribute_values: int
    total_entities: int
    total_filled_cells: int
    
    @property
    def yield_rate(self) -> float:
        """Proportion of new content added this query."""
        if self.total_filled_cells == 0:
            return 1.0
        return self.new_attribute_values / max(self.total_filled_cells, 1)


class Schemify:
    """
    AI-powered entity extraction and schema discovery system.
    
    Uses web search to ground extractions in real, citable sources.
    Employs a query queue to systematically explore attribute-value
    combinations for comprehensive entity discovery.
    
    Example:
        ```python
        from schemify import Schemify, SchemifyConfig
        
        config = SchemifyConfig(api_key="sk-...")
        schemify = Schemify(config)
        
        # Initialize (no API calls)
        await schemify.initialize(
            category="Open-source relational databases",
            guidance="Focus on widely deployed systems with active communities",
            schema_attributes=predefined_schema,  # optional
        )
        
        # Run exploration
        df = await schemify.run(max_queries=50)
        
        # Continue exploration
        df = await schemify.run(max_queries=20)
        ```
    """
    
    def __init__(self, config: SchemifyConfig, search_provider: SearchProvider | None = None):
        """
        Initialize Schemify.
        
        Args:
            config: Configuration options
            search_provider: Optional custom search provider (default: OpenAI web search)
        """
        self.config = config
        self.llm = LLMClient(config)
        
        # Initialize cache
        if config.cache_enabled:
            self.cache = Cache(ttl_hours=config.cache_ttl_hours)
        else:
            self.cache = NoOpCache()
        
        # Search provider (decoupled from LLM for swappable backends)
        self._search_provider = search_provider or OpenAISearchProvider(self.llm)
        
        # Initialize engines
        self.extraction = ExtractionEngine(
            config, self.llm, self.cache,
            search_provider=self._search_provider
        )
        self.resolution = ResolutionEngine(config, self.llm)
        
        # Current record set
        self.record_set: Optional[RecordSet] = None
        
        # Query queue for exploration
        self.query_queue: Optional[QueryQueue] = None
        
        # Query tracking
        self.query_history: list[IterationMetrics] = []
        self._query_counter: int = 0
        
        # Auto-complete settings
        self.auto_complete_threshold: float = 0.3  # Complete if >30% missing
        self.auto_expand_threshold: float = 0.5    # Expand if confidence <50%
        
        # Callbacks for progress updates
        self._on_progress: Optional[Callable[[str, int, int], None]] = None
        
        # Track initialization state
        self._initialized: bool = False
    
    def on_progress(self, callback: Callable[[str, int, int], None]):
        """
        Set a progress callback.
        
        Args:
            callback: Function(stage, current, total) called during operations
        """
        self._on_progress = callback
    
    def _report_progress(self, stage: str, current: int, total: int):
        """Report progress if callback is set."""
        if self._on_progress:
            self._on_progress(stage, current, total)
        logger.info(f"{stage}: {current}/{total}")
    
    def _display_table(self) -> pd.DataFrame:
        """Display the current records as a styled table and return DataFrame."""
        df = self.to_dataframe()
        if len(df) > 0:
            display(df)
        return df

    async def initialize(
        self,
        category: str,
        guidance: str = "",
        schema_attributes: list[SchemaAttribute] | None = None,
        suggest_schema: bool = True,
        generate_attribute_values: bool = True,
        cardinality_threshold: int = 50,
    ) -> "Schemify":
        """
        Initialize Schemify for a category. Sets up schema and query queue.
        
        This method performs setup only - no queries are executed.
        Call run() afterwards to execute discovery queries.
        
        Args:
            category: The type of entities to discover
            guidance: Natural language guidance for extraction
            schema_attributes: Predefined schema attributes (overrides suggest_schema).
                              Can include provisional_values and is_closed_set.
            suggest_schema: Whether to proactively suggest schema attributes
                           (requires 1 LLM call if no schema_attributes provided)
            generate_attribute_values: Whether to generate provisional attribute values
                                       for attributes that don't already have them
                                       (requires 1 LLM call per attribute without values)
            cardinality_threshold: Max values for closed set classification (default: 50)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Initializing Schemify for: {category}")
        
        # Initialize record set and reset tracking
        self.record_set = RecordSet(
            category=category,
            guidance=guidance,
        )
        self.query_history = []
        self._query_counter = 0
        
        # Use predefined schema if provided, otherwise suggest
        if schema_attributes:
            self.record_set.schema_attributes = schema_attributes
            logger.info(f"Using {len(schema_attributes)} predefined schema attributes")
        elif suggest_schema:
            self._report_progress("Suggesting schema", 0, 1)
            suggestions = await self.resolution.suggest_schema(category, guidance)
            self.record_set.schema_attributes = suggestions
            self._report_progress("Suggesting schema", 1, 1)
        
        # Generate provisional attribute values for attributes that don't have them
        if generate_attribute_values and self.record_set.schema_attributes:
            # Only generate for attributes without provisional values
            attrs_needing_values = [
                a for a in self.record_set.schema_attributes 
                if not a.provisional_values
            ]
            
            if attrs_needing_values:
                self._report_progress("Generating attribute values", 0, 1)
                await self.resolution.generate_attribute_values(
                    category=category,
                    guidance=guidance,
                    attributes=attrs_needing_values,
                    cardinality_threshold=cardinality_threshold,
                )
                self._report_progress("Generating attribute values", 1, 1)
            
            closed_attrs = [a for a in self.record_set.schema_attributes if a.is_closed_set]
            logger.info(f"Total closed-set attributes: {len(closed_attrs)}")
        
        # Initialize query queue
        self.query_queue = QueryQueue(seed=self.config.seed)
        
        # Populate queue with broad query and attribute-based queries
        self.query_queue.populate_from_attributes(
            self.record_set.schema_attributes,
            include_broad=True,
            lazy=self.config.lazy_generation,
        )
        
        logger.info(f"Initialized query queue: {self.query_queue.get_stats()}")
        self._initialized = True
        
        return self

    async def run_agentic(
        self,
        max_queries: int = 100,
        concurrency: int = 5,
        output_dir: str | None = None,
        seed_state: str | None = None,
        seed_records: str | list[dict] | pd.DataFrame | None = None,
        phase_split: tuple[float, float, float] = (0.60, 0.20, 0.20),
    ) -> pd.DataFrame:
        """
        Run hybrid phased agent-driven exploration.
        
        Three phases with structurally enforced budget splits:
          Phase 1 — Broad Discovery: Agent picks from generated search angles
            + its own creative queries. All parallel. No completions.
          Phase 2 — Targeted Discovery: Agent sees productive Phase 1 angles
            and generates intersectional queries. Light completions.
          Phase 3 — Completion: All completions run in PARALLEL. Light discovery.
        
        Requires initialize() to be called first.
        
        Args:
            max_queries: Total query budget (all phases)
            concurrency: Max parallel web searches
            output_dir: Where to save snapshots and logs
            seed_state: Path to a Schemify JSON save file to restore
            seed_records: Additional data to merge on top of seed_state
            phase_split: Budget fractions for (discovery, targeted, completion).
                Default (0.60, 0.20, 0.20) devotes 60% to broad discovery.
            
        Returns:
            DataFrame of all entities
        """
        if not self.record_set:
            raise ValueError("No record set. Call initialize() first.")
        
        from .strategy_agentic import AgenticStrategy
        
        strategy = AgenticStrategy(
            config=self.config,
            llm=self.llm,
            extraction=self.extraction,
            resolution=self.resolution,
        )
        
        history = await strategy.run(
            record_set=self.record_set,
            max_queries=max_queries,
            concurrency=concurrency,
            output_dir=output_dir,
            seed_state=seed_state,
            seed_records=seed_records,
            phase_split=phase_split,
        )
        
        # Post-processing (normalize attributes, display)
        await self._post_process()
        
        # Final export
        if output_dir:
            import os, json
            final_csv = os.path.join(output_dir, "final.csv")
            final_json = os.path.join(output_dir, "final.json")
            self.to_csv(final_csv, include_sources=True, include_evidence=True)
            self.save(final_json)
            self.llm.save_all_responses()
            usage = self.llm.get_usage_stats()
            with open(os.path.join(output_dir, "usage_stats.json"), "w") as f:
                json.dump(usage, f, indent=2)
        
        return self._display_table()

    async def verify_unverified(
        self,
        concurrency: int = 5,
        output_dir: str | None = None,
    ) -> dict:
        """
        Verify unverified attribute values via web search.

        For each entity with unsourced attribute values, makes one
        web-search call targeting those attributes.  Does NOT delete
        any values or entities.

        Args:
            concurrency: Max parallel web searches.
            output_dir: Where to write logs (reuses LLM output dir).

        Returns:
            Verification statistics dict.
        """
        if not self.record_set:
            raise ValueError("No record set. Call initialize() first.")

        from .strategy_agentic import AgenticStrategy

        strategy = AgenticStrategy(
            config=self.config,
            llm=self.llm,
            extraction=self.extraction,
            resolution=self.resolution,
        )
        return await strategy.verify_unverified_entities(
            self.record_set,
            concurrency=concurrency,
            output_dir=output_dir,
        )

    def finalize(
        self,
        output_dir: str | None = None,
    ) -> "RecordSet":
        """
        Build a finalized dataset of high-quality entities only.

        Entities with no web-sourced attributes are excluded.
        Unsourced attribute values within included entities are dropped.
        The original record_set is NOT modified.

        Args:
            output_dir: Where to save final.json and final.csv.

        Returns:
            A new RecordSet containing only verified entities/values.
        """
        if not self.record_set:
            raise ValueError("No record set. Call initialize() first.")

        from .strategy_agentic import AgenticStrategy

        strategy = AgenticStrategy(
            config=self.config,
            llm=self.llm,
            extraction=self.extraction,
            resolution=self.resolution,
        )
        return strategy.finalize(self.record_set, output_dir=output_dir)

    async def discover(
        self,
        category: str,
        guidance: str = "",
        max_queries: int = 20,
        schema_attributes: list[SchemaAttribute] | None = None,
        suggest_schema: bool = True,
        auto_complete: bool = True,
        generate_attribute_values: bool = True,
        cardinality_threshold: int = 50,
    ) -> pd.DataFrame:
        """
        Convenience method: Initialize and run discovery in one call.
        
        Equivalent to calling initialize() followed by run().
        
        For more control, use initialize() and run() separately:
            await schemify.initialize(category=..., schema_attributes=...)
            df = await schemify.run(max_queries=50)
        
        Args:
            category: The type of entities to discover
            guidance: Natural language guidance for extraction
            max_queries: Maximum discovery queries to execute (default: 20)
            schema_attributes: Predefined schema attributes (overrides suggest_schema)
            suggest_schema: Whether to proactively suggest schema attributes
            auto_complete: Whether to auto-complete missing values
            generate_attribute_values: Whether to generate provisional attribute values
            cardinality_threshold: Max values for closed set classification
            
        Returns:
            DataFrame of discovered entities (also displays table)
        """
        # Initialize
        await self.initialize(
            category=category,
            guidance=guidance,
            schema_attributes=schema_attributes,
            suggest_schema=suggest_schema,
            generate_attribute_values=generate_attribute_values,
            cardinality_threshold=cardinality_threshold,
        )
        
        # Run discovery
        return await self.run(
            max_discovery_queries=max_queries,
        )
    
    async def _process_query_queue(self, max_queries: int) -> list[IterationMetrics]:
        """
        Process queries from the queue up to max_queries.
        
        Each query execution:
        1. Gets the next priority query from queue
        2. Builds exclusion list from well-covered entities
        3. Executes web search and extraction
        4. Deduplicates results
        5. Adds new records
        6. May spawn completion queries for incomplete records
        """
        if not self.record_set or not self.query_queue:
            raise ValueError("No record set or query queue initialized")
        
        metrics_list = []
        consecutive_zero_yield = 0
        
        for i in range(max_queries):
            # Get next query
            query = self.query_queue.get_next_query()
            if not query:
                logger.info("Query queue exhausted")
                break
            
            self._report_progress("Processing queries", i + 1, max_queries)
            self._query_counter += 1
            
            # Snapshot before query
            entities_before = len(self.record_set.records)
            filled_before = self._count_filled_cells()
            
            # Execute the query
            new_records = await self._execute_query(query)
            
            # Check for duplicates (uses fuzzy matching + LLM)
            merged_count = 0
            if new_records and self.record_set.records:
                new_labels = [r.label for r in new_records]
                existing_labels = self.record_set.get_labels()
                # Build alias map for better matching
                alias_map = {r.label: r.aliases for r in self.record_set.records if r.aliases}
                duplicates = await self.resolution.check_duplicates(
                    new_labels, existing_labels,
                    use_fuzzy=True,
                    existing_aliases=alias_map,
                )
                # Merge duplicates into existing records instead of discarding
                for record in new_records:
                    if record.label in duplicates:
                        existing = self.record_set.get_record(duplicates[record.label])
                        if existing:
                            existing.merge_from(record)
                            merged_count += 1
                new_records = [r for r in new_records if r.label not in duplicates]
            
            # Add new records (with fuzzy matching for any remaining)
            for record in new_records:
                was_added, existing = self.record_set.add_record(record, use_fuzzy=True)
                if not was_added and existing:
                    merged_count += 1
            
            # Mark query as executed
            self.query_queue.mark_executed(query, entities_found=len(new_records))
            
            # Compute metrics
            entities_after = len(self.record_set.records)
            filled_after = self._count_filled_cells()
            
            metrics = IterationMetrics(
                query_num=self._query_counter,
                query_type=query.query_type.value,
                new_entities=entities_after - entities_before,
                new_attribute_values=filled_after - filled_before,
                total_entities=entities_after,
                total_filled_cells=filled_after,
            )
            metrics_list.append(metrics)
            self.query_history.append(metrics)
            
            logger.info(
                f"Query {self._query_counter} [{query.query_type.value}]: "
                f"+{metrics.new_entities} entities, "
                f"+{metrics.new_attribute_values} values "
                f"(yield: {metrics.yield_rate:.1%}, queue: {self.query_queue.pending_count()} pending)"
            )
        
        return metrics_list
    
    async def _execute_query(self, query: ExplorationQuery) -> list[Record]:
        """
        Execute a single exploration query.
        
        Args:
            query: The exploration query to execute
            
        Returns:
            List of new records discovered
        """
        if not self.record_set:
            return []
        
        # Build focus text based on query type
        if query.query_type == QueryType.DISCOVERY_BROAD:
            focus_text = ""
        elif query.query_type == QueryType.DISCOVERY_SINGLE:
            attr_name, attr_value = list(query.attribute_constraints.items())[0]
            focus_text = f'Find examples where {attr_name} is specifically "{attr_value}".'
        elif query.query_type == QueryType.DISCOVERY_PAIR:
            items = list(query.attribute_constraints.items())
            focus_text = (
                f'Find examples that combine:\n'
                f'- {items[0][0]}: "{items[0][1]}"\n'
                f'- {items[1][0]}: "{items[1][1]}"'
            )
        elif query.query_type == QueryType.DISCOVERY_TRIPLE:
            items = list(query.attribute_constraints.items())
            focus_text = (
                f'Find examples that combine ALL of:\n'
                f'- {items[0][0]}: "{items[0][1]}"\n'
                f'- {items[1][0]}: "{items[1][1]}"\n'
                f'- {items[2][0]}: "{items[2][1]}"'
            )
        elif query.query_type == QueryType.DISCOVERY_REFLECTIVE:
            # Reflective queries use custom_focus from LLM reflection
            focus_text = query.custom_focus or ""
        elif query.query_type == QueryType.COMPLETION:
            # Completion queries target specific records
            return await self._execute_completion_query(query)
        else:
            focus_text = ""
        
        # Execute discovery query
        new_records = await self.extraction.discover_entities(
            self.record_set,
            subcategory_focus=focus_text,
        )
        
        return new_records
    
    async def _execute_completion_query(self, query: ExplorationQuery) -> list[Record]:
        """Execute a completion query for a specific record."""
        if not self.record_set:
            return []
        
        # Find the target record
        target = None
        for record in self.record_set.records:
            if record.label == query.target_record_label:
                target = record
                break
        
        if not target:
            logger.warning(f"Target record not found: {query.target_record_label}")
            return []
        
        # Expand the record with target attributes
        await self.extraction.expand_record(
            target,
            self.record_set,
            target_attributes=query.target_attributes,
        )
        
        return []  # Completion queries don't add new records
    
    def _count_filled_cells(self) -> int:
        """Count the number of filled attribute cells."""
        if not self.record_set:
            return 0
        
        count = 0
        for record in self.record_set.records:
            for attr in record.attributes.values():
                if attr and attr.value:
                    count += 1
        return count
    
    def get_queue_stats(self):
        """
        Get statistics about the query queue.
        
        Returns:
            QueryQueueStats with:
            - pending/executed counts
            - by_type: Breakdown by query type
            - by_priority: Breakdown by priority
        """
        if not self.query_queue:
            return None
        return self.query_queue.get_stats()
    
    async def run(
        self,
        max_discovery_queries: int | None = None,
        max_completion_queries: int | None = None,
        output_dir: str | None = None,
        concurrency: int = 5,
        enable_reflection: bool | None = None,
        reflection_threshold: int | None = None,
        run_completion: bool | None = None,
    ) -> pd.DataFrame:
        """
        Process the query queue with level-parallel execution.
        
        Executes queries in parallel within each depth level, then prunes
        before moving to the next level. This provides full parallelism
        while preserving all pruning benefits.
        
        When reflection is enabled, the LLM analyzes current entities and
        query yields to generate strategic new queries when:
        - A level yields few new entities
        - The queue is exhausted but max_queries not reached
        
        Execution order:
        1. Level 0 (broad queries) - parallel
        2. Prune Level 1+ based on results
        3. Level 1 (singles) - parallel  
        4. Prune Level 2+ based on zero-yield singles
        5. Level 2 (pairs) - parallel
        6. Prune Level 3 based on zero-yield pairs
        7. Level 3 (triples) - parallel
        8. [If enabled] Reflect and add new queries when yield drops
        9. [If completion_enabled] Run completion pass to fill missing attributes
        
        Requires initialize() to be called first.
        
        Args:
            max_discovery_queries: Maximum discovery/exploration queries to run (default: entire queue)
            max_completion_queries: Maximum completion queries per entity (default: from config.max_completion_calls_per_entity)
            output_dir: Directory to save periodic snapshots and activity log
            concurrency: Maximum concurrent queries per level (default: 5)
            enable_reflection: Use LLM reflection for new queries (default: from config)
            reflection_threshold: Min entities before reflection triggers (default: from config)
            run_completion: Run completion pass after queries (default: from config.completion_enabled)
            
        Returns:
            DataFrame of all entities (also displays table)
        """
        # Use config defaults if not specified
        if enable_reflection is None:
            enable_reflection = self.config.enable_reflection
        if reflection_threshold is None:
            reflection_threshold = self.config.reflection_threshold
        if run_completion is None:
            run_completion = self.config.completion_enabled
        
        import os
        from datetime import datetime
        
        if not self.record_set:
            raise ValueError("No record set. Call initialize() first.")
        
        if not self.query_queue:
            raise ValueError("No query queue. Call initialize() first.")
        
        # Default to processing entire queue
        queue_stats = self.query_queue.get_stats()
        if max_discovery_queries is None:
            max_discovery_queries = queue_stats.pending_discovery
        
        # Set up output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Set output dir on LLM client for raw response logging
            self.llm.set_output_dir(output_dir)
            log_path = os.path.join(output_dir, "activity_log.txt")
            log_file = open(log_path, "a", encoding="utf-8")
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Schemify Run (level-parallel): {datetime.now().isoformat()}\n")
            log_file.write(f"Category: {self.record_set.category}\n")
            log_file.write(f"Max discovery queries: {max_discovery_queries}, Max completion queries: {max_completion_queries}, Concurrency: {concurrency}\n")
            log_file.write(f"Queue stats: {queue_stats}\n")
            log_file.write(f"{'='*60}\n\n")
        else:
            log_file = None
        
        try:
            queries_run = 0
            stop_reason = None
            reflection_count = 0
            max_reflections = self.config.max_reflections
            low_yield_threshold = self.config.reflection_low_yield
            
            # Process each depth level
            for depth in range(4):  # 0=broad, 1=singles, 2=pairs, 3=triples
                if queries_run >= max_discovery_queries:
                    stop_reason = f"max discovery queries ({max_discovery_queries})"
                    break
                
                # Early stop check: if last N queries all yielded 0 new entities
                early_stop_window = self.config.early_stop_window
                if (early_stop_window > 0
                    and len(self.query_history) >= early_stop_window
                    and queries_run >= early_stop_window):
                    recent = self.query_history[-early_stop_window:]
                    if all(m.new_entities == 0 for m in recent):
                        stop_reason = (
                            f"early stop (last {early_stop_window} queries yielded 0 new entities). "
                            f"Call run() again to continue."
                        )
                        logger.info(stop_reason)
                        if log_file:
                            log_file.write(f"\n  → {stop_reason}\n")
                        break
                
                # Lazy generation: generate pair/triple queries from productive results
                if self.config.lazy_generation and depth == 2:
                    productive = self.query_queue.get_productive_values(depth=1)
                    added = self.query_queue.generate_pairs_from_productive(
                        productive, self.record_set.schema_attributes
                    )
                    if log_file and added:
                        log_file.write(f"  → Lazy gen: {added} pair queries from productive singles\n")
                elif self.config.lazy_generation and depth == 3:
                    productive = self.query_queue.get_productive_values(depth=2)
                    added = self.query_queue.generate_triples_from_productive(
                        productive, self.record_set.schema_attributes
                    )
                    if log_file and added:
                        log_file.write(f"  → Lazy gen: {added} triple queries from productive pairs\n")
                
                # Get remaining budget for this level
                budget = max_discovery_queries - queries_run
                
                # Get queries at this depth
                level_queries = self.query_queue.take_queries_at_depth(depth, max_count=budget)
                
                if not level_queries:
                    continue
                
                level_name = ["broad", "singles", "pairs", "triples"][depth]
                level_display = ["Broad", "Single", "Pair", "Triple"][depth]
                logger.info(f"Level {depth} ({level_name}): {len(level_queries)} queries, concurrency={concurrency}")
                if log_file:
                    log_file.write(f"\n--- Level {depth} ({level_name}): {len(level_queries)} queries ---\n")
                
                # Snapshot entities before level
                entities_before_level = len(self.record_set.records)
                
                # Execute level in parallel batches
                level_results = await self._execute_level_parallel(
                    level_queries, 
                    concurrency=concurrency,
                    log_file=log_file,
                    level_name=level_display,
                )
                
                queries_run += len(level_queries)
                self._report_progress("Processing discovery", queries_run, max_discovery_queries)
                
                # Calculate level yield rate (entities per query)
                entities_after_level = len(self.record_set.records)
                level_new_entities = entities_after_level - entities_before_level
                level_yield_rate = level_new_entities / len(level_queries) if level_queries else 0
                
                # Incremental schema evolution after each level
                new_attrs = await self._evolve_schema_incrementally(log_file)
                
                # Promote open-set attributes with bounded cardinality to closed-set
                promoted = self._promote_emergent_closed_sets(log_file)
                
                # Execute any newly-generated queries at already-passed depth levels
                # (e.g., singles added after depth 1 has already run)
                if promoted and queries_run < max_discovery_queries:
                    for past_depth in range(depth + 1):
                        backfill = self.query_queue.take_queries_at_depth(
                            past_depth, max_count=max_discovery_queries - queries_run
                        )
                        if backfill:
                            past_name = ["Broad", "Single", "Pair", "Triple"][past_depth]
                            logger.info(f"Backfill: {len(backfill)} {past_name.lower()} queries from promoted attributes")
                            if log_file:
                                log_file.write(f"  → Backfill: {len(backfill)} {past_name.lower()} queries from emergent closed-sets\n")
                            backfill_results = await self._execute_level_parallel(
                                backfill,
                                concurrency=concurrency,
                                log_file=log_file,
                                level_name=f"{past_name}+",
                            )
                            queries_run += len(backfill)
                
                # Export snapshot after each level
                if output_dir:
                    csv_path = os.path.join(output_dir, f"level_{depth}_{level_name}.csv")
                    self.to_csv(csv_path, include_sources=True, include_evidence=True)
                    if log_file:
                        log_file.write(f"  → Saved level snapshot: {csv_path}\n")
                        log_file.write(f"  → Level yield: {level_new_entities} entities from {len(level_queries)} queries ({level_yield_rate:.1%})\n")
                
                # Reflection: trigger if level yield rate was low and we have budget
                # low_yield_threshold is now interpreted as a percentage (e.g., 0.1 = 10%)
                if (enable_reflection 
                    and reflection_count < max_reflections
                    and queries_run < max_discovery_queries
                    and len(self.record_set.records) >= reflection_threshold
                    and level_yield_rate < low_yield_threshold
                    and depth >= 1):  # Don't reflect after broad
                    
                    logger.info(f"Low yield rate ({level_yield_rate:.1%}) at level {depth}, triggering reflection...")
                    if log_file:
                        log_file.write(f"\n  → Low yield rate ({level_yield_rate:.1%}), triggering reflection...\n")
                    
                    suggestions = await self.reflect(add_to_queue=True, max_queries=3)
                    reflection_count += 1
                    
                    if suggestions and log_file:
                        log_file.write(f"  → Added {len(suggestions)} reflective queries\n")
                        for s in suggestions:
                            log_file.write(f"    - {s.get('focus', '')[:60]}...\n")
            
            # After all depth levels, check if we should reflect on exhaustion
            while (queries_run < max_discovery_queries 
                   and enable_reflection 
                   and reflection_count < max_reflections):
                
                # Check for any remaining queries
                remaining = self.query_queue.pending_count()
                
                if remaining > 0:
                    # Execute remaining queries (including reflective ones)
                    budget = min(max_discovery_queries - queries_run, remaining)
                    for _ in range(budget):
                        query = self.query_queue.get_next_query()
                        if not query:
                            break
                        try:
                            await self._execute_and_record_query(query, log_file)
                        except Exception as e:
                            logger.error(f"Query failed (post-reflection): {e}")
                            if log_file:
                                log_file.write(f"  ERROR (post-reflection): {e}\n")
                        queries_run += 1
                else:
                    # Queue exhausted but budget remains - reflect for more queries
                    if len(self.record_set.records) >= reflection_threshold:
                        logger.info(f"Queue exhausted with budget remaining, triggering reflection...")
                        if log_file:
                            log_file.write(f"\n  → Queue exhausted, reflecting for more queries...\n")
                        
                        suggestions = await self.reflect(add_to_queue=True, max_queries=5)
                        reflection_count += 1
                        
                        if not suggestions:
                            stop_reason = "reflection returned no new queries"
                            break
                        
                        if log_file:
                            log_file.write(f"  → Added {len(suggestions)} reflective queries\n")
                    else:
                        stop_reason = "queue exhausted (not enough entities for reflection)"
                        break
            
            if not stop_reason:
                stop_reason = "complete"
            
            final_stats = self.query_queue.get_stats()
            logger.info(f"Stopped after {queries_run} queries: {stop_reason}")
            if log_file:
                log_file.write(f"\nStopped: {stop_reason}\n")
                log_file.write(f"Final queue stats: {final_stats}\n")
                if final_stats.poisoned_values:
                    log_file.write(f"Poisoned values: {final_stats.poisoned_values}\n")
            
            # Export structured activity log
            if output_dir:
                activity_json = os.path.join(output_dir, "activity_log.json")
                self.query_queue.export_activity_log(activity_json, format="json")
            
            # Run completion pass AFTER all queries complete
            if run_completion:
                logger.info("Running completion pass for missing attributes...")
                if log_file:
                    log_file.write("\n--- Completion Pass ---\n")
                # Use max_completion_queries if provided, otherwise use config default
                calls_per_entity = max_completion_queries if max_completion_queries is not None else self.config.max_completion_calls_per_entity
                if log_file:
                    log_file.write(f"  Max completion queries per entity: {calls_per_entity}\n")
                completion_stats = await self.complete_all(
                    concurrency=concurrency,
                    verbose=False,
                    max_calls_per_entity=calls_per_entity,
                )
                if log_file:
                    log_file.write(f"  Entities processed: {completion_stats['entities_processed']}\n")
                    log_file.write(f"  Total completion calls: {completion_stats['total_calls']}\n")
                    log_file.write(f"  Attributes filled: {completion_stats['attributes_filled']}\n")
                    log_file.write(f"  Still missing: {completion_stats['still_missing_count']}\n")
            
            # Post-processing and display
            await self._post_process()
            
            # Final export
            if output_dir:
                final_csv = os.path.join(output_dir, "final.csv")
                final_json = os.path.join(output_dir, "final.json")
                self.to_csv(final_csv, include_sources=True, include_evidence=True)
                self.save(final_json)
                
                # Save all raw API responses
                self.llm.save_all_responses()
                
                # Save usage stats
                usage_stats = self.llm.get_usage_stats()
                usage_path = os.path.join(output_dir, "usage_stats.json")
                with open(usage_path, 'w') as f:
                    import json
                    json.dump(usage_stats, f, indent=2)
                
                if log_file:
                    log_file.write(f"\nFinal export: {final_csv}, {final_json}\n")
                    log_file.write(f"Total records: {len(self.record_set.records)}\n")
                    log_file.write(f"Schema attributes: {[a.name for a in self.record_set.schema_attributes]}\n")
                    log_file.write(f"\nAPI Usage:\n")
                    log_file.write(f"  Input tokens: {usage_stats['input_tokens']:,}\n")
                    log_file.write(f"  Output tokens: {usage_stats['output_tokens']:,}\n")
                    log_file.write(f"  Total tokens: {usage_stats['total_tokens']:,}\n")
                    log_file.write(f"  Web searches: {usage_stats['web_search_count']}\n")
                    log_file.write(f"  Total API calls: {usage_stats['total_api_calls']}\n")
            
            return self._display_table()
        
        finally:
            if log_file:
                log_file.close()
    
    async def _execute_level_parallel(
        self,
        queries: list,
        concurrency: int,
        log_file=None,
        level_name: str = "",
    ) -> list[dict]:
        """
        Execute a batch of queries in parallel with rate limiting.
        
        Args:
            queries: List of ExplorationQuery objects
            concurrency: Max concurrent executions
            log_file: Optional file handle for logging
            level_name: Name of the level for progress context (e.g., "Broad", "Singles")
            
        Returns:
            List of result dicts with query outcomes
        """
        from datetime import datetime
        
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        results_lock = asyncio.Lock()
        total_queries = len(queries)
        query_index = [0]  # Use list for mutable closure
        cumulative_new = [0]  # Track new entities across level
        cumulative_values = [0]  # Track new values across level
        
        async def execute_one(query):
            async with semaphore:
                # Set progress context before executing
                query_index[0] += 1
                idx = query_index[0]
                
                # Build context with running totals
                total_entities = len(self.record_set.records) if self.record_set else 0
                total_values = self._count_filled_cells()
                context = f"{level_name} {idx}/{total_queries} | {total_entities} entities, {total_values} values"
                self.llm.set_progress_context(context)
                
                try:
                    result = await self._execute_and_record_query(query, log_file=None)
                except Exception as e:
                    logger.error(f"Query failed ({level_name} {idx}/{total_queries}): {e}")
                    if log_file:
                        log_file.write(f"  ERROR query {idx}: {e}\n")
                    result = {
                        "query_id": getattr(query, 'id', None),
                        "new_entities": 0,
                        "new_values": 0,
                        "duplicates": 0,
                        "total_found": 0,
                        "error": str(e),
                    }
                
                # Update cumulative counts
                async with results_lock:
                    results.append(result)
                    cumulative_new[0] += result.get("new_entities", 0)
                    cumulative_values[0] += result.get("new_values", 0)
                return result
        
        # Execute all in parallel
        await asyncio.gather(*[execute_one(q) for q in queries])
        
        # Clear progress context
        self.llm.set_progress_context("")
        
        # Log summary with totals
        total_new = sum(r.get("new_entities", 0) for r in results)
        total_new_values = sum(r.get("new_values", 0) for r in results)
        total_dups = sum(r.get("duplicates", 0) for r in results)
        total_entities = len(self.record_set.records) if self.record_set else 0
        total_values = self._count_filled_cells()
        
        logger.info(
            f"  Level complete: {len(queries)} queries → "
            f"+{total_new} entities, +{total_new_values} values ({total_dups} dups) | "
            f"Total: {total_entities} entities, {total_values} values"
        )
        if log_file:
            log_file.write(
                f"  Level complete: +{total_new} entities, +{total_new_values} values ({total_dups} dups) | "
                f"Total: {total_entities} entities, {total_values} values\n"
            )
        
        return results
    
    async def _execute_and_record_query(self, query, log_file=None) -> dict:
        """
        Execute a single query and record results.
        
        Returns dict with execution results.
        """
        from datetime import datetime
        
        # These should never be None when this is called, but satisfy type checker
        assert self.record_set is not None
        assert self.query_queue is not None
        
        self._query_counter += 1
        
        # Snapshot before query
        entities_before = len(self.record_set.records)
        filled_before = self._count_filled_cells()
        
        # Execute query (let exceptions propagate to caller for isolation)
        new_records = await self._execute_query(query)
        total_found = len(new_records) if new_records else 0
        
        # Deduplicate (with fuzzy matching and merging)
        duplicates_count = 0
        if new_records and self.record_set.records:
            new_labels = [r.label for r in new_records]
            existing_labels = self.record_set.get_labels()
            # Build alias map for better matching
            alias_map = {r.label: r.aliases for r in self.record_set.records if r.aliases}
            duplicates = await self.resolution.check_duplicates(
                new_labels, existing_labels,
                use_fuzzy=True,
                existing_aliases=alias_map,
            )
            # Merge duplicates into existing records
            for record in new_records:
                if record.label in duplicates:
                    existing = self.record_set.get_record(duplicates[record.label])
                    if existing:
                        existing.merge_from(record)
                        duplicates_count += 1
            new_records = [r for r in new_records if r.label not in duplicates]
        
        # Add records (with fuzzy matching for remaining)
        for record in new_records:
            was_added, existing = self.record_set.add_record(record, use_fuzzy=True)
            if not was_added and existing:
                duplicates_count += 1
        
        # Update provisional values
        self._update_provisional_values(new_records)
        
        # Mark executed
        new_entities_count = len(new_records)
        self.query_queue.mark_executed(
            query, 
            entities_found=total_found,
            new_entities=new_entities_count,
        )
        
        # Compute metrics
        entities_after = len(self.record_set.records)
        filled_after = self._count_filled_cells()
        
        metrics = IterationMetrics(
            query_num=self._query_counter,
            query_type=query.query_type.value,
            new_entities=entities_after - entities_before,
            new_attribute_values=filled_after - filled_before,
            total_entities=entities_after,
            total_filled_cells=filled_after,
        )
        self.query_history.append(metrics)
        
        # Log
        stats = self.query_queue.get_stats()
        log_msg = (
            f"Query {metrics.query_num} [{metrics.query_type}]: "
            f"+{metrics.new_entities} new ({duplicates_count} dups)"
        )
        logger.debug(log_msg)
        if log_file:
            log_file.write(f"{datetime.now().strftime('%H:%M:%S')} | {log_msg}\n")
        
        return {
            "query_id": query.id,
            "new_entities": new_entities_count,
            "new_values": metrics.new_attribute_values,
            "duplicates": duplicates_count,
            "total_found": total_found,
        }
    
    async def reflect(
        self,
        add_to_queue: bool = True,
        max_queries: int = 5,
    ) -> list[dict]:
        """
        Use LLM reflection to analyze current state and generate new strategic queries.
        
        The LLM examines:
        - Entities discovered so far (summaries of key attributes)
        - Query history with yields (what worked, what didn't)
        - Attribute value coverage (gaps in the data)
        
        Then generates new broad queries targeting underexplored areas.
        
        Args:
            add_to_queue: If True, add generated queries to the queue
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of suggested queries with focus, rationale, and expected entities
        """
        if not self.record_set or len(self.record_set.records) < 3:
            logger.warning("Not enough entities for meaningful reflection (need at least 3)")
            return []
        
        # Build entity summaries (condensed for token efficiency)
        entity_summaries = self._build_entity_summaries(max_entities=30)
        
        # Build query history summary
        query_history = self._build_query_history_summary()
        
        # Build coverage summary
        coverage_summary = self._build_coverage_summary()
        
        # Call LLM for reflection
        result = await self.llm.structured_completion(
            prompt=prompts.QUERY_REFLECTION,
            response_format=prompts.REFLECTION_QUERY_SCHEMA,
            variables={
                "category": self.record_set.category,
                "guidance": self.record_set.guidance or "",
                "entity_count": len(self.record_set.records),
                "entity_summaries": entity_summaries,
                "query_count": len(self.query_history),
                "query_history": query_history,
                "coverage_summary": coverage_summary,
            }
        )
        
        analysis = result.get("analysis", "")
        suggested = result.get("suggested_queries", [])[:max_queries]
        
        logger.info(f"Reflection analysis: {analysis}")
        logger.info(f"Generated {len(suggested)} reflective queries")
        
        # Add to queue if requested
        if add_to_queue and suggested and self.query_queue:
            for i, query_data in enumerate(suggested):
                priority = {
                    "high": QueryPriority.HIGH,
                    "medium": QueryPriority.MEDIUM,
                    "low": QueryPriority.LOW,
                }.get(query_data.get("priority", "medium"), QueryPriority.MEDIUM)
                
                query = ExplorationQuery(
                    id=f"reflect_{self._query_counter + i + 1}",
                    query_type=QueryType.DISCOVERY_REFLECTIVE,
                    priority=priority,
                    custom_focus=query_data.get("focus", ""),
                )
                self.query_queue.add_query(query)
                logger.info(f"Added reflective query: {query_data.get('focus', '')[:60]}...")
        
        return suggested
    
    def _build_entity_summaries(self, max_entities: int = 30) -> str:
        """Build condensed entity summaries for reflection prompt."""
        if not self.record_set:
            return "(no entities yet)"
        
        lines = []
        records = self.record_set.records[:max_entities]
        
        for record in records:
            # Get key attributes (closed-set ones for categorization)
            attrs = []
            for attr in self.record_set.schema_attributes:
                if attr.is_closed_set:
                    val = record.attributes.get(attr.name)
                    if val and val.value:
                        attrs.append(f"{attr.name}: {val.value}")
            
            attr_str = "; ".join(attrs) if attrs else "(no attributes)"
            lines.append(f"- {record.label}: {attr_str}")
        
        if len(self.record_set.records) > max_entities:
            lines.append(f"... and {len(self.record_set.records) - max_entities} more entities")
        
        return "\n".join(lines)
    
    def _build_query_history_summary(self) -> str:
        """Build query history summary for reflection prompt."""
        if not self.query_history:
            return "(no queries executed yet)"
        
        lines = []
        
        # Group by query type
        by_type: dict[str, list[IterationMetrics]] = {}
        for m in self.query_history:
            by_type.setdefault(m.query_type, []).append(m)
        
        for query_type, metrics in by_type.items():
            total_entities = sum(m.new_entities for m in metrics)
            zero_yield = sum(1 for m in metrics if m.new_entities == 0)
            lines.append(
                f"- {query_type}: {len(metrics)} queries, "
                f"{total_entities} entities found, "
                f"{zero_yield} zero-yield"
            )
        
        # Show best and worst performing queries
        if len(self.query_history) > 5:
            sorted_queries = sorted(self.query_history, key=lambda m: m.new_entities, reverse=True)
            lines.append("\nTop performers:")
            for m in sorted_queries[:3]:
                lines.append(f"  - Query {m.query_num} [{m.query_type}]: +{m.new_entities} entities")
        
        return "\n".join(lines)
    
    def _build_coverage_summary(self) -> str:
        """Build attribute value coverage summary for reflection prompt."""
        if not self.record_set:
            return "(no coverage data)"
        
        lines = []
        
        for attr in self.record_set.schema_attributes:
            if not attr.is_closed_set or not attr.provisional_values:
                continue
            
            # Count entities per value
            value_counts: dict[str, int] = {}
            for record in self.record_set.records:
                attr_val = record.attributes.get(attr.name)
                if attr_val and attr_val.values:
                    for sv in attr_val.values:
                        value_counts[sv.value] = value_counts.get(sv.value, 0) + 1
            
            if not value_counts:
                lines.append(f"\n{attr.name}: (no values found)")
                continue
            
            # Find underrepresented values
            total = sum(value_counts.values())
            avg = total / len(attr.provisional_values) if attr.provisional_values else 0
            
            underrepresented = [
                v for v in attr.provisional_values
                if value_counts.get(v, 0) < avg * 0.5
            ]
            
            lines.append(f"\n{attr.name}:")
            # Show distribution
            for val, count in sorted(value_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {val}: {count} entities")
            
            if underrepresented:
                lines.append(f"  Underrepresented: {', '.join(underrepresented[:5])}")
        
        return "\n".join(lines) if lines else "(no closed-set attributes)"

    async def _evolve_schema_incrementally(self, log_file=None) -> list[SchemaAttribute]:
        """
        Evolve schema after a batch of queries and add new queries for discovered attributes.
        
        This enables dynamic schema expansion during extraction:
        1. Check for new frequent attributes in additional_attributes
        2. Promote them to schema
        3. Generate provisional values (if they look like closed-set)
        4. Add queries for the new attribute
        
        Returns:
            List of newly promoted attributes
        """
        if not self.record_set or not self.query_queue:
            return []
        
        # Need enough records to make meaningful decisions
        if len(self.record_set.records) < 5:
            return []
        
        # Evolve schema - returns newly promoted attributes
        new_attrs = self.resolution.evolve_schema(self.record_set)
        
        if not new_attrs:
            return []
        
        # For each new attribute, try to determine if it's closed-set
        # and generate provisional values
        for attr in new_attrs:
            # Collect observed values from records
            observed_values: set[str] = set()
            for record in self.record_set.records:
                attr_val = record.attributes.get(attr.name)
                if attr_val and attr_val.value:
                    # Handle multi-value attributes
                    for val in attr_val.value.split("|"):
                        val = val.strip()
                        if val and len(val) < 100:  # Skip long values (probably not categorical)
                            observed_values.add(val)
            
            # If we have reasonable cardinality, treat as closed-set
            if 2 <= len(observed_values) <= 50:
                attr.provisional_values = list(observed_values)
                attr.is_closed_set = True
                
                # Generate queries for this new attribute
                existing_closed = [
                    a for a in self.record_set.schema_attributes 
                    if a.is_closed_set and a.name != attr.name
                ]
                queries_added = self.query_queue.add_queries_for_new_attribute(
                    attr, 
                    existing_closed,
                    include_singles=True,
                    include_pairs=True,
                    include_triples=False,
                )
                
                logger.info(f"New schema attribute '{attr.name}': {len(observed_values)} values, {queries_added} queries added")
                if log_file:
                    log_file.write(f"  Schema evolution: '{attr.name}' promoted with {len(observed_values)} values, {queries_added} new queries\n")
            else:
                attr.is_closed_set = False
                logger.info(f"New schema attribute '{attr.name}': open-set (cardinality: {len(observed_values)})")
                if log_file:
                    log_file.write(f"  Schema evolution: '{attr.name}' promoted as open-set\n")
        
        # Expand enums for closed-set attributes where too many entities got "Other"
        if self.record_set and len(self.record_set.records) >= 5:
            expansions = await self.resolution.expand_enum_values(self.record_set)
            if expansions and log_file:
                for attr_name, new_vals in expansions.items():
                    log_file.write(f"  Enum expansion: '{attr_name}' added {len(new_vals)} values: {new_vals}\n")
        
        return new_attrs
    
    def _promote_emergent_closed_sets(self, log_file=None) -> list[SchemaAttribute]:
        """
        Promote open-set attributes to closed-set when observed values suggest
        a bounded categorical dimension.
        
        This enables proactive query generation from schema that emerges during
        discovery, without requiring a predefined schema. Values may be
        renamed/merged/split later during normalization — that's fine, the
        goal here is exploration coverage.
        
        Criteria for promotion:
        - Attribute is currently open-set (is_closed_set=False)
        - Has accumulated 2-50 distinct provisional values from records
        - Has been observed across enough records (>=3) to be meaningful
        
        Returns:
            List of newly promoted attributes
        """
        if not self.record_set or not self.query_queue:
            return []
        
        if len(self.record_set.records) < 5:
            return []
        
        promoted = []
        
        for attr in self.record_set.schema_attributes:
            if attr.is_closed_set:
                continue  # Already closed
            
            # Collect observed values from records
            observed_values: set[str] = set()
            records_with_attr = 0
            for record in self.record_set.records:
                attr_val = record.attributes.get(attr.name)
                if attr_val and attr_val.value:
                    records_with_attr += 1
                    for val in attr_val.value.split("|"):
                        val = val.strip()
                        if val and len(val) < 100:
                            observed_values.add(val)
            
            # Need enough records to be confident this is categorical
            if records_with_attr < 3:
                continue
            
            # Check cardinality sweet spot: bounded but not trivial
            if 2 <= len(observed_values) <= attr.cardinality_threshold:
                attr.is_closed_set = True
                attr.provisional_values = list(observed_values)
                
                # Generate queries for the newly-closed attribute
                existing_closed = [
                    a for a in self.record_set.schema_attributes
                    if a.is_closed_set and a.name != attr.name
                ]
                queries_added = self.query_queue.add_queries_for_new_attribute(
                    attr,
                    existing_closed,
                    include_singles=True,
                    include_pairs=True,
                    include_triples=False,
                )
                
                promoted.append(attr)
                logger.info(
                    f"Emergent closed-set: '{attr.name}' promoted with "
                    f"{len(observed_values)} values, {queries_added} queries added"
                )
                if log_file:
                    log_file.write(
                        f"  Emergent closed-set: '{attr.name}' → "
                        f"{len(observed_values)} values, {queries_added} new queries\n"
                    )
        
        return promoted
    
    async def _auto_complete_if_needed(self):
        """Auto-complete records if missing value ratio exceeds threshold."""
        if not self.record_set or not self.record_set.records:
            return
        
        # Calculate missing value ratio
        schema_attrs = {a.name for a in self.record_set.schema_attributes}
        if not schema_attrs:
            return
        
        total_cells = len(self.record_set.records) * len(schema_attrs)
        missing_cells = 0
        
        for record in self.record_set.records:
            for attr_name in schema_attrs:
                attr = record.attributes.get(attr_name)
                if not attr or not attr.value:
                    missing_cells += 1
        
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        if missing_ratio >= self.auto_complete_threshold:
            logger.info(f"Auto-completing: {missing_ratio:.1%} missing values")
            await self.extraction.complete_missing_values(self.record_set)
    
    async def complete_all(
        self,
        concurrency: int = 3,
        verbose: bool = True,
        max_calls_per_entity: int | None = None,
    ) -> dict:
        """
        Complete missing attributes for all entities.
        
        For each entity with missing attributes, makes up to max_calls_per_entity
        LLM calls, each time prompting ONLY for the attributes that are still missing.
        Any additional values returned (for attributes that were already populated)
        are still saved/merged.
        
        This runs AFTER the main query loop completes, not during.
        
        Args:
            concurrency: Max concurrent completion calls (default: 3)
            verbose: Whether to print progress (default: True)
            max_calls_per_entity: Max LLM calls per entity (default: from config)
            
        Returns:
            Dict with completion statistics:
            - entities_processed: Number of entities that had missing attrs
            - total_calls: Total LLM completion calls made
            - attributes_filled: Number of attributes that got filled
            - still_missing_count: Count of still-missing attributes
        """
        if not self.record_set:
            raise ValueError("No record set. Call initialize() and run() first.")
        
        record_set = self.record_set  # Type narrowing for closures
        
        if max_calls_per_entity is None:
            max_calls_per_entity = self.config.max_completion_calls_per_entity
        
        schema_attrs = {a.name for a in record_set.schema_attributes}
        
        def get_missing_attrs(record: Record) -> list[str]:
            """Get list of attribute names that have no value."""
            return [a for a in schema_attrs if a not in record.attributes or not record.attributes[a].value]
        
        # Find all records with ANY missing attributes
        incomplete = [(r, get_missing_attrs(r)) for r in record_set.records if get_missing_attrs(r)]
        entities_before = len(incomplete)
        
        if verbose:
            print(f"=== Completion Pass ===\n")
            print(f"Entities with missing attributes: {entities_before}/{len(record_set.records)}")
            print(f"Max calls per entity: {max_calls_per_entity}")
        
        if not incomplete:
            if verbose:
                print("All entities complete!")
            return {
                "entities_processed": 0,
                "total_calls": 0,
                "attributes_filled": 0,
                "still_missing_count": 0,
            }
        
        # Track statistics
        total_calls = 0
        attrs_filled_before = sum(
            1 for r in record_set.records 
            for a in schema_attrs 
            if a in r.attributes and r.attributes[a].value
        )
        
        # Process each entity with missing attributes
        semaphore = asyncio.Semaphore(concurrency)
        entity_calls: dict[str, int] = {}  # Track calls per entity
        entity_index = [0]  # For progress tracking
        total_entities = len(incomplete)
        
        async def complete_entity(record: Record, missing_attrs: list[str]):
            """Complete a single entity, up to max_calls_per_entity."""
            nonlocal total_calls
            entity_id = record.label
            entity_calls[entity_id] = 0
            
            # Track entity index
            entity_index[0] += 1
            entity_num = entity_index[0]
            
            for call_num in range(max_calls_per_entity):
                # Get current missing attributes (may have changed from prior calls)
                current_missing = get_missing_attrs(record)
                
                if not current_missing:
                    # All attributes filled - done with this entity
                    logger.debug(f"Entity {entity_id} complete after {call_num} calls")
                    break
                
                async with semaphore:
                    try:
                        # Set progress context with running totals
                        current_filled = self._count_filled_cells()
                        context = f"Completion {entity_num}/{total_entities} | {current_filled} values, {len(current_missing)} missing"
                        self.llm.set_progress_context(context)
                        
                        # Call expand_record targeting ONLY missing attributes
                        # Any additional values returned will still be saved
                        await self.extraction.expand_record(
                            record,
                            record_set,
                            target_attributes=current_missing,
                        )
                        entity_calls[entity_id] += 1
                        total_calls += 1
                        
                    except Exception as e:
                        logger.error(f"Error completing {entity_id} (call {call_num+1}): {e}")
                        break
            
            if verbose and entity_calls[entity_id] > 0:
                final_missing = get_missing_attrs(record)
                if final_missing:
                    logger.debug(f"  {entity_id}: {entity_calls[entity_id]} calls, still missing: {final_missing}")
                else:
                    logger.debug(f"  {entity_id}: complete after {entity_calls[entity_id]} calls")
        
        if verbose:
            print(f"\nProcessing {len(incomplete)} entities...")
            for record, missing in incomplete[:5]:
                print(f"  {record.label}: missing {len(missing)} attrs ({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''})")
            if len(incomplete) > 5:
                print(f"  ... and {len(incomplete) - 5} more")
        
        # Run completion for all incomplete entities
        logger.info(f"Running completion for {len(incomplete)} entities (max {max_calls_per_entity} calls each)")
        tasks = [complete_entity(r, m) for r, m in incomplete]
        await asyncio.gather(*tasks)
        
        # Clear progress context
        self.llm.set_progress_context("")
        
        # Calculate final statistics
        attrs_filled_after = sum(
            1 for r in record_set.records 
            for a in schema_attrs 
            if a in r.attributes and r.attributes[a].value
        )
        attributes_filled = attrs_filled_after - attrs_filled_before
        
        still_missing = [
            (r.label, get_missing_attrs(r)) 
            for r in record_set.records 
            if get_missing_attrs(r)
        ]
        still_missing_count = sum(len(m) for _, m in still_missing)
        
        if verbose:
            print(f"\n=== Completion Result ===")
            print(f"Total LLM calls: {total_calls}")
            print(f"Attributes filled: {attributes_filled}")
            if still_missing:
                print(f"Still missing: {still_missing_count} attribute values across {len(still_missing)} entities")
                for label, missing in still_missing[:5]:
                    print(f"  {label}: {', '.join(missing)}")
                if len(still_missing) > 5:
                    print(f"  ... and {len(still_missing) - 5} more entities")
            else:
                print("All entities fully complete!")
        
        return {
            "entities_processed": entities_before,
            "total_calls": total_calls,
            "attributes_filled": attributes_filled,
            "still_missing_count": still_missing_count,
        }
    
    def _update_provisional_values(self, new_records: list[Record]):
        """
        Update schema provisional values from discovered entities.
        
        When we discover new entities, their attribute values may include
        values not in the original provisional_values list. Adding these
        expands future exploration possibilities.
        
        Tracks values for ALL attributes (not just closed-set), so that
        open-set attributes accumulate observed values for potential
        later promotion to closed-set via _promote_emergent_closed_sets.
        """
        if not self.record_set or not new_records:
            return
        
        for attr in self.record_set.schema_attributes:
            existing_values = set(attr.provisional_values)
            
            for record in new_records:
                attr_value = record.attributes.get(attr.name)
                if attr_value and attr_value.value:
                    # Split on common delimiters (some values are lists)
                    values = [v.strip() for v in attr_value.value.split("|")]
                    for value in values:
                        if value and len(value) < 100 and value not in existing_values:
                            # Add new value to provisional values
                            attr.provisional_values.append(value)
                            existing_values.add(value)
                            logger.debug(f"Added provisional value: {attr.name}={value}")
    
    async def _auto_expand_low_confidence(self):
        """Auto-expand records with low confidence scores."""
        if not self.record_set:
            return
        
        low_conf_records = [
            r for r in self.record_set.records
            if r.average_confidence() < self.auto_expand_threshold
        ]
        
        if low_conf_records:
            logger.info(f"Auto-expanding {len(low_conf_records)} low-confidence records")
            for record in low_conf_records:
                await self.extraction.expand_record(record, self.record_set)

    async def expand(
        self,
        target_labels: list[str] | None = None,
        target_attributes: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Expand existing records with additional information.
        
        Args:
            target_labels: Specific records to expand (default: all)
            target_attributes: Specific attributes to search for
            
        Returns:
            Updated DataFrame (also displays table)
            Updated RecordSet
        """
        if not self.record_set:
            raise ValueError("No record set. Call discover() first.")
        
        records_to_expand = self.record_set.records
        if target_labels:
            records_to_expand = [
                r for r in self.record_set.records
                if r.label in target_labels
            ]
        
        total = len(records_to_expand)
        for i, record in enumerate(records_to_expand):
            self._report_progress("Expanding records", i + 1, total)
            await self.extraction.expand_record(
                record,
                self.record_set,
                target_attributes=target_attributes,
            )
        
        await self._post_process()
        return self._display_table()
    
    async def complete(self) -> pd.DataFrame:
        """
        Complete missing attribute values across all records.
        
        Returns:
            Updated DataFrame (also displays table)
        """
        if not self.record_set:
            raise ValueError("No record set. Call discover() first.")
        
        self._report_progress("Completing records", 0, 1)
        await self.extraction.complete_missing_values(self.record_set)
        self._report_progress("Completing records", 1, 1)
        
        await self._post_process()
        return self._display_table()
    
    async def _post_process(self):
        """Run post-processing steps after extraction."""
        if not self.record_set:
            return
        
        # Merge near-duplicate schema attributes
        await self.resolution.merge_similar_attributes(self.record_set)
        
        # Deterministic key cleanup: promote additional→core when names match
        # case-insensitively, and collapse case-variant duplicates within
        # additional_attributes. Runs before the LLM pass so it sees less noise.
        self.resolution.consolidate_attribute_keys(self.record_set)

        # Resolve attribute names
        await self.resolution.resolve_attribute_names(self.record_set)
        
        # Normalize attribute values for closed-set attributes
        await self.resolution.normalize_attribute_values(self.record_set)
        
        # Clean up poorly formatted values (e.g., "April 2014" -> "2014")
        await self.resolution.cleanup_value_formats(
            self.record_set,
            attribute_formats={
                "Year Founded": "4-digit year like 2014",
                "Year Initiated": "4-digit year like 2014", 
                "Year Launched": "4-digit year like 2014",
                "Launch Date": "4-digit year like 2014",
            }
        )
        
        # Fix inconsistent Title Case from LLM extraction (sentence case)
        self.resolution.fix_value_capitalization(self.record_set)
        
        # Evolve schema based on frequencies
        self.resolution.evolve_schema(self.record_set)
        
        # Update frequencies
        self.record_set.update_schema_frequencies()

        # Deterministic finalization: ALL-CAPS labels, fold "Also known as"
        # into aliases, and lift units into numeric attribute names.
        self.resolution.finalize_normalization(self.record_set)
    
    async def cleanup_formats(self, attribute_formats: dict[str, str]) -> None:
        """
        Clean up poorly formatted attribute values using LLM.
        
        Use this to standardize values that have inconsistent formats,
        e.g., "April 2014", "Apr-14", "2014Q1" all becoming "2014".
        
        Args:
            attribute_formats: Mapping of attribute names to expected format descriptions.
                Example: {"Year Founded": "4-digit year like 2014"}
        """
        if not self.record_set:
            return
        await self.resolution.cleanup_value_formats(self.record_set, attribute_formats)
    
    async def normalize(
        self,
        attributes: list[str] | None = None,
        fuzzy_threshold: int = 75,
        cardinality_threshold: int = 50,
    ) -> dict[str, dict]:
        """
        Run an automatic normalization pass over the record set.
        
        For each attribute, analyzes the distribution of observed values,
        clusters near-duplicates, classifies the attribute as open or closed
        based on the data, and maps all values to canonical forms.
        
        Args:
            attributes: Specific attribute names to normalize (default: all schema attributes)
            fuzzy_threshold: Similarity threshold for clustering (0-100, default: 75)
            cardinality_threshold: Max canonical values to still be considered closed-set
            
        Returns:
            Dict mapping attribute_name -> {
                "classification": "closed" | "open",
                "unique_raw": number of distinct raw values,
                "unique_clustered": effective groups after clustering,
                "canonical_values": list of canonical values,
                "mappings": dict of original -> canonical,
                "reasoning": LLM explanation,
            }
        """
        if not self.record_set:
            raise ValueError("No record set. Call initialize() first.")
        return await self.resolution.auto_normalize(
            self.record_set,
            attributes=attributes,
            fuzzy_threshold=fuzzy_threshold,
            cardinality_threshold=cardinality_threshold,
        )

    def to_dataframe(
        self, 
        include_sources: bool = False,
        include_evidence: bool = False,
        collapse_additional: bool = True,
    ) -> pd.DataFrame:
        """
        Export current records to a pandas DataFrame.
        
        Args:
            include_sources: Whether to include per-attribute source URLs
            include_evidence: Whether to include per-attribute evidence snippets
                            (the actual text from sources that supports each value)
            collapse_additional: Whether to collapse non-schema attributes into
                               a single "Additional Attributes" column (default: True)
            
        Returns:
            DataFrame with one row per record
        """
        if not self.record_set or not self.record_set.records:
            return pd.DataFrame()
        
        rows = []
        for record in self.record_set.records:
            row = record.to_flat_dict(
                include_sources=include_sources,
                include_evidence=include_evidence,
                collapse_additional=collapse_additional,
            )
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def to_csv(
        self, 
        filepath: str, 
        include_sources: bool = False,
        include_evidence: bool = False,
        collapse_additional: bool = True,
    ):
        """
        Export current records to a CSV file.
        
        Args:
            filepath: Path to save the CSV
            include_sources: Whether to include source URLs
            include_evidence: Whether to include per-attribute evidence snippets
            collapse_additional: Whether to collapse non-schema attributes
        """
        df = self.to_dataframe(
            include_sources=include_sources,
            include_evidence=include_evidence,
            collapse_additional=collapse_additional,
        )
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} records to {filepath}")
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the current extraction."""
        stats = {
            "llm_usage": self.llm.get_usage_stats(),
            "cache": self.cache.get_stats(),
        }
        
        if self.record_set:
            stats["records"] = {
                "total": len(self.record_set.records),
                "schema_attributes": len(self.record_set.schema_attributes),
                "category": self.record_set.category,
            }
            
            # Confidence distribution
            confidences = [r.average_confidence() for r in self.record_set.records]
            if confidences:
                stats["records"]["avg_confidence"] = sum(confidences) / len(confidences)
                stats["records"]["min_confidence"] = min(confidences)
                stats["records"]["max_confidence"] = max(confidences)
        
        return stats
    
    def get_low_confidence_records(self, threshold: float = 0.5) -> list[Record]:
        """
        Get records with confidence below threshold.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            List of low-confidence records
        """
        if not self.record_set:
            return []
        
        return [
            r for r in self.record_set.records
            if r.average_confidence() < threshold
        ]
    
    def save(self, filepath: str):
        """
        Save the current record set to a JSON file.
        
        Args:
            filepath: Path to save the JSON
        """
        if not self.record_set:
            raise ValueError("No record set to save")
        
        with open(filepath, "w") as f:
            f.write(self.record_set.to_json())
        
        logger.info(f"Saved record set to {filepath}")
    
    def export_dashboard(self, filepath: str) -> int:
        """
        Export a slim dashboard_data.js for local file:// loading.

        Strips internal metadata (source tiers, timestamps, additional
        attributes, confidence scores) to produce a minified JS file
        that the HTML dashboards can load via ``<script src>``.

        Args:
            filepath: Destination path (e.g. ``output/dashboard_data.js``).

        Returns:
            File size in bytes.
        """
        if not self.record_set:
            raise ValueError("No record set to export")
        from .strategy_agentic import export_dashboard_js
        return export_dashboard_js(self.record_set, filepath)

    def load(self, filepath: str) -> RecordSet:
        """
        Load a record set from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Loaded RecordSet
        """
        with open(filepath, "r") as f:
            self.record_set = RecordSet.from_json(f.read())
        
        logger.info(f"Loaded {len(self.record_set.records)} records from {filepath}")
        return self.record_set
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
    
    async def stream(
        self,
        max_discovery_queries: int | None = None,
        concurrency: int = 5,
    ):
        """
        Stream discovery results as an async generator.
        
        Yields events as records are discovered, enabling incremental display
        and early termination.
        
        Yields:
            Tuples of (event_type, data):
            - ("record", Record) — new record discovered
            - ("progress", dict) — progress update with stats
            - ("level_complete", dict) — a depth level finished
            - ("early_stop", dict) — early stop triggered
            - ("complete", dict) — all done
        
        Example:
            async for event_type, data in schemify.stream(max_discovery_queries=50):
                if event_type == "record":
                    print(f"Found: {data.label}")
                elif event_type == "early_stop":
                    break  # or continue if you want to keep going
        """
        if not self.record_set or not self.query_queue:
            raise ValueError("Not initialized. Call initialize() first.")
        
        queue_stats = self.query_queue.get_stats()
        if max_discovery_queries is None:
            max_discovery_queries = queue_stats.pending_discovery
        
        queries_run = 0
        
        for depth in range(4):
            if queries_run >= max_discovery_queries:
                break
            
            # Early stop check
            early_stop_window = self.config.early_stop_window
            if (early_stop_window > 0
                and len(self.query_history) >= early_stop_window
                and queries_run >= early_stop_window):
                recent = self.query_history[-early_stop_window:]
                if all(m.new_entities == 0 for m in recent):
                    yield ("early_stop", {
                        "reason": f"Last {early_stop_window} queries yielded 0 new entities",
                        "queries_run": queries_run,
                        "total_entities": len(self.record_set.records),
                    })
                    return
            
            # Lazy generation
            if self.config.lazy_generation and depth == 2:
                productive = self.query_queue.get_productive_values(depth=1)
                self.query_queue.generate_pairs_from_productive(
                    productive, self.record_set.schema_attributes
                )
            elif self.config.lazy_generation and depth == 3:
                productive = self.query_queue.get_productive_values(depth=2)
                self.query_queue.generate_triples_from_productive(
                    productive, self.record_set.schema_attributes
                )
            
            budget = max_discovery_queries - queries_run
            level_queries = self.query_queue.take_queries_at_depth(depth, max_count=budget)
            if not level_queries:
                continue
            
            entities_before = len(self.record_set.records)
            
            # Execute queries sequentially for streaming (parallel within semaphore)
            semaphore = asyncio.Semaphore(concurrency)
            
            async def execute_and_collect(query):
                async with semaphore:
                    return await self._execute_and_record_query(query)
            
            for query in level_queries:
                records_before = set(r.label for r in self.record_set.records)
                await self._execute_and_record_query(query)
                queries_run += 1
                
                # Yield any new records
                for record in self.record_set.records:
                    if record.label not in records_before:
                        yield ("record", record)
                
                # Yield progress
                yield ("progress", {
                    "queries_run": queries_run,
                    "max_queries": max_discovery_queries,
                    "total_entities": len(self.record_set.records),
                    "total_values": self._count_filled_cells(),
                    "depth": depth,
                })
            
            level_name = ["broad", "singles", "pairs", "triples"][depth]
            entities_after = len(self.record_set.records)
            yield ("level_complete", {
                "depth": depth,
                "level": level_name,
                "queries": len(level_queries),
                "new_entities": entities_after - entities_before,
                "total_entities": entities_after,
            })
        
        yield ("complete", {
            "queries_run": queries_run,
            "total_entities": len(self.record_set.records),
            "total_values": self._count_filled_cells(),
        })
