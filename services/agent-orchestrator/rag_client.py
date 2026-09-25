"""
Aegis — Resilient RAG Client
------------------------------
The orchestrator depends on the RAG service being reachable, but network
calls fail. This client wraps every request in:

  1. Exponential-backoff retries (bounded, jittered) for transient failures
  2. A circuit breaker that trips after repeated failures, so a struggling
     RAG service doesn't cascade into slow/hanging orchestrator calls
  3. A safe fallback (empty result set) so agents keep functioning —
     alerts are still produced, just without a cited protocol source

This is the same reliability pattern the frontend's "SYSTEM RELIABILITY"
panel visualizes in the demo (simulated there since GitHub Pages can't
run this service) — here it's the real implementation.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

import httpx
import os

logger = logging.getLogger("rag-client")

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8003")
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.2
FAILURE_THRESHOLD = 5          # consecutive failures before the breaker trips
OPEN_COOLDOWN_SECONDS = 30      # how long the breaker stays open before testing again


class BreakerState(Enum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # failing fast, not calling the service
    HALF_OPEN = "half_open" # testing whether the service has recovered


@dataclass
class CircuitBreaker:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = field(default=0.0)

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = BreakerState.CLOSED

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= FAILURE_THRESHOLD:
            self.state = BreakerState.OPEN
            self.opened_at = time.time()
            logger.warning("Circuit breaker OPEN after %d consecutive failures", self.consecutive_failures)

    def allow_request(self) -> bool:
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.OPEN:
            if time.time() - self.opened_at >= OPEN_COOLDOWN_SECONDS:
                self.state = BreakerState.HALF_OPEN
                logger.info("Circuit breaker HALF_OPEN — testing service recovery")
                return True
            return False
        return True  # HALF_OPEN: allow exactly one probe request


_breaker = CircuitBreaker()


class RAGClient:
    def __init__(self, base_url: str = RAG_SERVICE_URL, timeout: float = 5.0):
        self.base_url = base_url
        self.timeout = timeout

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        if not _breaker.allow_request():
            logger.warning("Circuit breaker OPEN — short-circuiting RAG call, returning empty context")
            return []

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/v1/retrieve",
                    json={"query": query, "k": k},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                _breaker.record_success()
                return resp.json()["results"]
            except httpx.HTTPError as exc:
                last_exc = exc
                _breaker.record_failure()
                if attempt < MAX_RETRIES:
                    backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    backoff += random.uniform(0, backoff * 0.3)  # jitter
                    logger.warning(
                        "RAG retrieve failed (attempt %d/%d): %s — retrying in %.2fs",
                        attempt, MAX_RETRIES, exc, backoff,
                    )
                    time.sleep(backoff)

        logger.error("RAG retrieve exhausted retries: %s", last_exc)
        return []
