"""
Caching layer for Schemify.

Provides TTL-based caching for web search results and extractions
to reduce API costs and enable offline operation.
"""

import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger("schemify.cache")


class Cache:
    """
    SQLite-based cache with TTL support.
    """
    
    def __init__(self, cache_dir: str = ".schemify_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.db_path = self.cache_dir / "cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TEXT,
                    expires_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
            """)
            conn.commit()
    
    def _make_key(self, operation: str, **kwargs) -> str:
        """Generate a cache key from operation and parameters."""
        data = json.dumps({"operation": operation, **kwargs}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get(self, operation: str, **kwargs) -> Optional[Any]:
        """
        Get a cached value if it exists and hasn't expired.
        
        Args:
            operation: The type of operation (e.g., "web_search", "extraction")
            **kwargs: Parameters that identify this specific request
            
        Returns:
            Cached value or None if not found/expired
        """
        key = self._make_key(operation, **kwargs)
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
                (key, now)
            )
            row = cursor.fetchone()
            
            if row:
                logger.debug(f"Cache hit for {operation}")
                return json.loads(row[0])
            
            logger.debug(f"Cache miss for {operation}")
            return None
    
    def set(self, value: Any, operation: str, **kwargs):
        """
        Cache a value.
        
        Args:
            value: The value to cache (must be JSON-serializable)
            operation: The type of operation
            **kwargs: Parameters that identify this specific request
        """
        key = self._make_key(operation, **kwargs)
        now = datetime.now()
        expires_at = now + self.ttl
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, json.dumps(value), now.isoformat(), expires_at.isoformat())
            )
            conn.commit()
        
        logger.debug(f"Cached {operation} (expires: {expires_at})")
    
    def invalidate(self, operation: str = None, **kwargs):
        """
        Invalidate cached entries.
        
        Args:
            operation: If provided, only invalidate entries for this operation
            **kwargs: If provided with operation, invalidate specific entry
        """
        with sqlite3.connect(self.db_path) as conn:
            if operation and kwargs:
                key = self._make_key(operation, **kwargs)
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            else:
                # Invalidate all expired entries
                now = datetime.now().isoformat()
                conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
            conn.commit()
    
    def clear(self):
        """Clear all cached entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            valid = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at > ?", (now,)
            ).fetchone()[0]
            expired = total - valid
            
        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": expired,
            "cache_path": str(self.db_path),
        }


class NoOpCache:
    """A cache that doesn't cache (for when caching is disabled)."""
    
    def get(self, operation: str, **kwargs) -> None:
        return None
    
    def set(self, value: Any, operation: str, **kwargs):
        pass
    
    def invalidate(self, operation: str = None, **kwargs):
        pass
    
    def clear(self):
        pass
    
    def get_stats(self) -> dict[str, Any]:
        return {"enabled": False}
