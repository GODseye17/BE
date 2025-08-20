"""
Connection Pool for HTTP requests
"""
import asyncio
from typing import Optional
import aiohttp
import logging

logger = logging.getLogger(__name__)

class ConnectionPool:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent connections
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def request(self, method: str, url: str, **kwargs):
        """Make HTTP request with connection pooling"""
        async with self.semaphore:
            session = await self.get_session()
            try:
                async with session.request(method, url, **kwargs) as response:
                    return await response.json()
            except Exception as e:
                logger.error(f"HTTP request failed: {e}")
                raise
