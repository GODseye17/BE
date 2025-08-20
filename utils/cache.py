"""
Cache Manager for Redis-based caching
"""
import redis
import json
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    def set(self, key: str, value: Any, expire: int = 3600):
        """Set cached value with expiration"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(key, expire, json.dumps(value))
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache by pattern"""
        if not self.redis_client:
            return
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
