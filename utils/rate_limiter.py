"""
Rate Limiter for API request throttling
"""
import asyncio
import time
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        
        return False
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for a client"""
        now = time.time()
        
        if client_id not in self.requests:
            return self.max_requests
        
        # Count recent requests
        recent_requests = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]
        
        return max(0, self.max_requests - len(recent_requests))
