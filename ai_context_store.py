# ai_context_store.py
# In-memory store for live laptop AI vision context.
# The laptop AI POSTs its context here, and the orchestrator reads it when reasoning.

import time
import logging

logger = logging.getLogger(__name__)


class AIContextStore:
    """
    Holds the latest vision/sensor context from the Laptop AI.
    Updated by POST /director/ai/context.
    Read by CloudOrchestrator when processing AI commands.
    """

    def __init__(self):
        self._context = {}
        self._last_update = 0

    def update(self, data: dict):
        """Store latest laptop AI context."""
        self._context = data
        self._last_update = time.time()
        logger.info(f"📡 Laptop AI Context Updated ({len(data)} keys, age=0s)")

    def get_context(self) -> dict:
        """Get latest context. Returns empty dict if stale (>30s old)."""
        age = time.time() - self._last_update
        if age > 30 and self._last_update > 0:
            logger.warning(f"⚠️ Laptop AI context is stale ({age:.0f}s old)")
            return {"_stale": True, "_age_seconds": age, **self._context}
        return self._context

    def get_age(self) -> float:
        if self._last_update == 0:
            return -1  # Never received
        return time.time() - self._last_update


# Singleton instance
context_store = AIContextStore()
