"""
LLM client for Schemify with web search support.

Supports:
- gpt-5.2 with Responses API for web search and reasoning control
- Comprehensive usage tracking (input/output tokens, web searches)
- Raw response logging for debugging and data preservation
"""

from openai import AsyncOpenAI
from datetime import datetime
from typing import Any, Optional
import asyncio
import json
import logging
import os
import random as _random

from .models import Citation, SchemifyConfig, BudgetExceededError

logger = logging.getLogger("schemify.llm")


# Approximate token pricing (input, output) per 1M tokens
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.0, 8.0),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.2": (2.0, 8.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),
    "o3": (2.0, 8.0),
    "o4-mini": (1.10, 4.40),
}

# Web search tool surcharge per call (USD)
# Current pricing (see https://developers.openai.com/api/docs/pricing):
# all models $10/1K calls; search content tokens billed at model rates.
WEB_SEARCH_CALL_COST: dict[str, float] = {
    "gpt-4o": 0.010,
    "gpt-4o-mini": 0.010,
    "gpt-4.1": 0.010,
    "gpt-4.1-mini": 0.010,
    "gpt-4.1-nano": 0.010,
    "gpt-5.2": 0.010,
    "gpt-5.4": 0.010,
    "gpt-5.4-mini": 0.010,
    "o3": 0.010,
    "o3-mini": 0.010,
    "o4-mini": 0.010,
}
DEFAULT_WEB_SEARCH_CALL_COST = 0.010  # Default for unknown models


class LLMClient:
    """
    LLM client with web search and structured output support.
    Uses the Responses API for reasoning control.
    Tracks all usage and saves raw responses.
    """
    
    def __init__(self, config: SchemifyConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=300.0,       # 5-minute timeout per request (web search + reasoning needs headroom)
            max_retries=0,       # We handle retries ourselves
        )
        
        # Detailed usage tracking
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.web_search_count = 0
        self.structured_completion_count = 0
        self.simple_completion_count = 0
        self.total_cost = 0.0
        
        # Raw response logging
        self.output_dir: Optional[str] = None
        self.query_count = 0
        self.raw_responses: list[dict] = []
        
        # Progress context for logging (set by caller)
        self._progress_context: str = ""
    
    def set_progress_context(self, context: str):
        """Set the current progress context for log messages."""
        self._progress_context = context
    
    def _log_with_context(self, message: str):
        """Log a message with the current progress context prefix."""
        if self._progress_context:
            logger.info(f"[{self._progress_context}] {message}")
        else:
            logger.info(message)
    
    def set_output_dir(self, output_dir: str):
        """Set the output directory for saving raw responses."""
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(os.path.join(output_dir, "raw_responses"), exist_ok=True)
    
    def _save_raw_response(self, query_type: str, prompt: str, response: Any, metadata: dict = None):
        """Save raw API response to JSON file."""
        self.query_count += 1
        
        # Build response record
        record = {
            "query_num": self.query_count,
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "model": self.config.search_model if query_type == "web_search" else self.config.completion_model,
            "prompt": prompt,
            "metadata": metadata or {},
        }
        
        # Extract response data
        try:
            # Convert response to dict-like structure
            response_data = {
                "id": getattr(response, 'id', None),
                "model": getattr(response, 'model', None),
                "created_at": getattr(response, 'created_at', None),
            }
            
            # Extract usage
            if response.usage:
                response_data["usage"] = {
                    "input_tokens": getattr(response.usage, 'input_tokens', 0),
                    "output_tokens": getattr(response.usage, 'output_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }
            
            # Extract output items
            output_items = []
            for item in response.output:
                item_data = {"type": item.type}
                
                if item.type == "web_search_call":
                    item_data["id"] = getattr(item, 'id', None)
                    item_data["status"] = getattr(item, 'status', None)
                
                elif item.type == "web_search_result":
                    # Capture the full web search results
                    results = []
                    if hasattr(item, 'results'):
                        for r in item.results:
                            results.append({
                                "url": getattr(r, 'url', ''),
                                "title": getattr(r, 'title', ''),
                                "snippet": getattr(r, 'snippet', ''),
                            })
                    item_data["results"] = results
                
                elif item.type == "message":
                    content_list = []
                    for content_item in item.content:
                        content_data = {"type": content_item.type}
                        if content_item.type == "output_text":
                            content_data["text"] = content_item.text
                            # Capture annotations
                            if hasattr(content_item, 'annotations') and content_item.annotations:
                                annotations = []
                                for ann in content_item.annotations:
                                    ann_data = {"type": ann.type}
                                    if ann.type == "url_citation":
                                        ann_data["url"] = getattr(ann, 'url', '')
                                        ann_data["title"] = getattr(ann, 'title', '')
                                        ann_data["start_index"] = getattr(ann, 'start_index', None)
                                        ann_data["end_index"] = getattr(ann, 'end_index', None)
                                    annotations.append(ann_data)
                                content_data["annotations"] = annotations
                        content_list.append(content_data)
                    item_data["content"] = content_list
                
                output_items.append(item_data)
            
            response_data["output"] = output_items
            record["response"] = response_data
            
        except Exception as e:
            logger.warning(f"Error extracting response data: {e}")
            record["response"] = {"error": str(e)}
        
        # Store in memory
        self.raw_responses.append(record)
        
        # Save to file if output_dir is set
        if self.output_dir:
            filepath = os.path.join(
                self.output_dir, 
                "raw_responses", 
                f"query_{self.query_count:04d}_{query_type}.json"
            )
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(record, f, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                logger.warning(f"Error saving raw response: {e}")
    
    def _track_usage(self, response: Any, is_web_search: bool = False):
        """Track token usage from response and update cost estimate."""
        if response.usage:
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
            total = getattr(response.usage, 'total_tokens', 0)
            
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += total
            
            if is_web_search:
                self.web_search_count += 1
            
            # Estimate cost
            model = self.config.search_model if is_web_search else self.config.completion_model
            input_price, output_price = MODEL_PRICING.get(model, (2.0, 8.0))
            call_cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
            
            # Add per-call web search tool surcharge
            if is_web_search:
                call_cost += WEB_SEARCH_CALL_COST.get(model, DEFAULT_WEB_SEARCH_CALL_COST)
            
            self.total_cost += call_cost
            
            # Check budget
            if self.config.max_budget is not None and self.total_cost > self.config.max_budget:
                raise BudgetExceededError(
                    f"Estimated cost ${self.total_cost:.4f} exceeds budget ${self.config.max_budget:.2f}"
                )
    
    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if an error is retryable (rate limit, server error, timeout)."""
        error_str = str(error).lower()
        return any(s in error_str for s in [
            'rate_limit', '429', '500', '502', '503', '504',
            'timeout', 'connection', 'overloaded',
        ])
    
    async def _retry_api_call(self, coro, max_retries: int = 5):
        """Execute an async API call with exponential backoff retry.
        
        Uses up to max_retries attempts with exponential backoff.
        Timeout and overload errors get longer waits (up to 120s).
        """
        for attempt in range(max_retries + 1):
            try:
                return await coro()
            except BudgetExceededError:
                raise  # Never retry budget errors
            except Exception as e:
                if attempt == max_retries or not self._is_retryable(e):
                    raise
                error_str = str(e).lower()
                is_timeout = 'timeout' in error_str
                # Longer backoff for timeouts/overload; shorter for rate limits
                if is_timeout or 'overloaded' in error_str:
                    delay = min(5.0 * (2 ** attempt) + _random.uniform(0, 3), 120.0)
                else:
                    delay = min(1.0 * (2 ** attempt) + _random.uniform(0, 1), 60.0)
                self._log_with_context(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
    
    async def web_search_completion(
        self,
        prompt: str,
        variables: dict[str, Any] = None,
        user_location: dict[str, str] = None,
    ) -> tuple[str, list[Citation]]:
        """
        Perform a web search completion using Responses API.
        
        Returns:
            Tuple of (response_text, citations)
        """
        variables = variables or {}
        formatted_prompt = prompt.format(**variables)
        
        logger.debug(f"Web search query: {formatted_prompt[:200]}...")
        
        # Build tools for web search
        tools = [{"type": "web_search"}]
        if user_location:
            tools = [{
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "approximate": user_location,
                }
            }]
        
        call_kwargs = {
            "model": self.config.search_model,
            "input": formatted_prompt,
            "tools": tools,
            "reasoning": {"effort": self.config.reasoning_effort},
        }
        if self.config.reasoning_effort == "none":
            call_kwargs["temperature"] = self.config.temperature
        response = await self._retry_api_call(lambda: self.client.responses.create(**call_kwargs))
        
        # Track usage
        self._track_usage(response, is_web_search=True)
        
        # Save raw response
        self._save_raw_response(
            "web_search",
            formatted_prompt,
            response,
            {"user_location": user_location}
        )
        
        # Extract text content from output items
        content = ""
        citations = []
        
        for item in response.output:
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        content = content_item.text
                        
                        # Extract citations from annotations
                        if hasattr(content_item, 'annotations') and content_item.annotations:
                            for annotation in content_item.annotations:
                                if annotation.type == "url_citation":
                                    url = annotation.url
                                    title = annotation.title
                                    start_idx = annotation.start_index
                                    end_idx = annotation.end_index
                                    
                                    # Extract evidence snippet
                                    snippet = None
                                    if start_idx is not None and content:
                                        text_before = content[:start_idx]
                                        import re
                                        sentences = re.split(r'(?<=[.!?])\s+', text_before)
                                        if sentences:
                                            if len(sentences) >= 2:
                                                snippet = ' '.join(sentences[-2:]).strip()
                                            else:
                                                snippet = sentences[-1].strip()
                                            snippet = snippet.replace('**', '').replace('*', '')
                                            snippet = re.sub(r'\(\[.*?\]\(.*?\)\)', '', snippet).strip()
                                            # Strip citation references like Source [2], [7], ([20])
                                            snippet = re.sub(r'Sources?\s*\[\d+(?:,\s*\d+)*\]\s*', '', snippet)
                                            snippet = re.sub(r'\(\[\d+(?:,\s*\d+)*\]\)', '', snippet)
                                            snippet = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', snippet)
                                            snippet = re.sub(r'\s{2,}', ' ', snippet).strip()
                                            # Capitalize first letter of each sentence after stripping
                                            snippet = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), snippet)
                                    
                                    citations.append(Citation(
                                        url=url,
                                        title=title,
                                        retrieved_at=datetime.now(),
                                        start_index=start_idx,
                                        end_index=end_idx,
                                        snippet=snippet,
                                    ))
        
        self._log_with_context(f"Web search returned {len(citations)} citations")
        return content, citations
    
    async def structured_completion(
        self,
        prompt: str,
        response_format: dict[str, Any],
        variables: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Perform a structured completion with JSON schema using Responses API.
        
        Returns:
            Parsed JSON response
        """
        variables = variables or {}
        formatted_prompt = prompt.format(**variables)
        
        logger.debug(f"Structured completion: {formatted_prompt[:200]}...")
        
        # Build kwargs for the API call
        kwargs = {
            "model": self.config.completion_model,
            "input": formatted_prompt,
            "reasoning": {"effort": self.config.reasoning_effort},
        }
        if self.config.reasoning_effort == "none":
            kwargs["temperature"] = self.config.temperature
        
        # Add structured output format if provided
        # The Responses API expects: text.format.type, text.format.name, text.format.schema
        if response_format and "json_schema" in response_format:
            json_schema = response_format["json_schema"]
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.get("name", "response"),
                    "schema": json_schema.get("schema", {}),
                    "strict": json_schema.get("strict", True),
                }
            }
        
        response = await self._retry_api_call(lambda: self.client.responses.create(**kwargs))
        
        # Track usage
        self._track_usage(response)
        self.structured_completion_count += 1
        
        # Save raw response
        self._save_raw_response(
            "structured_completion",
            formatted_prompt,
            response,
            {"schema_name": response_format.get("json_schema", {}).get("name", "unknown")}
        )
        
        # Extract text content
        content = ""
        for item in response.output:
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        content = content_item.text
                        break
        
        logger.debug(f"Structured response: {content[:500]}...")
        
        return json.loads(content)
    
    async def simple_completion(
        self,
        prompt: str,
        variables: dict[str, Any] = None,
    ) -> str:
        """
        Perform a simple text completion using Responses API.
        
        Returns:
            Response text
        """
        variables = variables or {}
        formatted_prompt = prompt.format(**variables)
        
        simple_kwargs = {
            "model": self.config.completion_model,
            "input": formatted_prompt,
            "reasoning": {"effort": self.config.reasoning_effort},
        }
        if self.config.reasoning_effort == "none":
            simple_kwargs["temperature"] = self.config.temperature
        response = await self._retry_api_call(lambda: self.client.responses.create(**simple_kwargs))
        
        # Track usage
        self._track_usage(response)
        self.simple_completion_count += 1
        
        # Save raw response
        self._save_raw_response("simple_completion", formatted_prompt, response)
        
        # Extract text content
        for item in response.output:
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        return content_item.text
        
        return ""
    
    def get_usage_stats(self) -> dict[str, Any]:
        """Get comprehensive usage statistics."""
        search_model = self.config.search_model
        web_search_surcharge = self.web_search_count * WEB_SEARCH_CALL_COST.get(
            search_model, DEFAULT_WEB_SEARCH_CALL_COST
        )
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "web_search_count": self.web_search_count,
            "web_search_surcharge": web_search_surcharge,
            "structured_completion_count": self.structured_completion_count,
            "simple_completion_count": self.simple_completion_count,
            "total_api_calls": self.query_count,
            "estimated_cost": self.total_cost,
        }
    
    def save_all_responses(self, filepath: str = None):
        """Save all raw responses to a single JSON file."""
        if filepath is None and self.output_dir:
            filepath = os.path.join(self.output_dir, "all_raw_responses.json")
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "usage_stats": self.get_usage_stats(),
                    "responses": self.raw_responses,
                }, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Saved {len(self.raw_responses)} raw responses to {filepath}")
