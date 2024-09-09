import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

class BaseServiceClient(ABC):
    def __init__(self, base_url: str, api_key: str, max_retries: int = 3, rate_limit: int = 100):
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.session = aiohttp.ClientSession()
        self.queue = asyncio.Queue()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    @abstractmethod
    async def request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass

    async def _make_request(self, method: str, url: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(method, url, json=data, headers=self._get_headers()) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Request failed with status {response.status}: {await response.text()}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except aiohttp.ClientError as e:
                logger.error(f"Client error: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        raise Exception("Max retries reached, request failed")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def enqueue_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None):
        await self.queue.put((method, endpoint, data))

    async def process_queue(self):
        while True:
            method, endpoint, data = await self.queue.get()
            try:
                response = await self.request(method, endpoint, data)
                logger.info(f"Request to {endpoint} succeeded with response: {response}")
            except Exception as e:
                logger.error(f"Request to {endpoint} failed: {e}")
            finally:
                self.queue.task_done()

    async def start_queue_processor(self):
        asyncio.create_task(self.process_queue())
