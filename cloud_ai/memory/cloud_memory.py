"""
cloud_ai/memory/cloud_memory.py — Persistent AI memory for the cloud orchestrator.

Stores and retrieves:
- Flight session history (what shots were taken, what worked/failed)
- User preferences (preferred shot styles, camera settings, subjects)
- Location knowledge (known obstacles, good angles, lighting conditions)
- Cinematic patterns (successful shot sequences for different scenarios)

This memory persists across drone sessions and informs the AI's decisions.
Stored as JSON files on the server's filesystem (simple, no DB dependency).

Data Flow:
    Orchestrator ──▶ CloudMemory.store() ──▶ JSON files on disk
    Orchestrator ◀── CloudMemory.recall() ◀── JSON files on disk
"""
import json
import os
import time
import logging
from typing import Optional

logger = logging.getLogger('cloud_memory')

# Default storage directory
MEMORY_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'memory')


class CloudMemory:
    """
    Persistent memory store for the AI orchestrator.

    Usage:
        memory = CloudMemory()

        # Store something
        memory.store('session', 'flight_001', {
            'shots': [...],
            'conditions': 'outdoor_sunset',
            'user_feedback': 'liked the orbit shot',
        })

        # Recall it
        data = memory.recall('session', 'flight_001')

        # Search for relevant memories
        results = memory.search('session', {'conditions': 'outdoor_sunset'})
    """

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or MEMORY_DIR
        os.makedirs(self.base_dir, exist_ok=True)

        # Memory categories (each gets its own subdirectory)
        self.categories = [
            'sessions',       # Flight session logs
            'preferences',    # User preferences and settings
            'locations',      # Location-specific knowledge
            'patterns',       # Successful cinematic patterns
            'feedback',       # User feedback on AI decisions
        ]
        for cat in self.categories:
            os.makedirs(os.path.join(self.base_dir, cat), exist_ok=True)

    def store(self, category: str, key: str, data: dict):
        """
        Store a memory entry.

        Args:
            category: One of 'sessions', 'preferences', 'locations', 'patterns', 'feedback'
            key: Unique identifier (e.g., 'flight_001', 'user_default_settings')
            data: Dict of data to store
        """
        if category not in self.categories:
            logger.warning(f"Unknown memory category: {category}")
            return False

        # Add metadata
        entry = {
            'key': key,
            'category': category,
            'timestamp': time.time(),
            'data': data,
        }

        filepath = os.path.join(self.base_dir, category, f"{key}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(entry, f, indent=2, default=str)
            logger.info(f"Memory stored: {category}/{key}")
            return True
        except Exception as e:
            logger.error(f"Memory store failed: {e}")
            return False

    def recall(self, category: str, key: str) -> Optional[dict]:
        """
        Recall a specific memory entry.
        Returns the data dict, or None if not found.
        """
        filepath = os.path.join(self.base_dir, category, f"{key}.json")
        try:
            with open(filepath, 'r') as f:
                entry = json.load(f)
            return entry.get('data')
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Memory recall failed: {e}")
            return None

    def search(self, category: str, filters: dict = None, limit: int = 10) -> list:
        """
        Search memories in a category.

        Args:
            category: Category to search
            filters: Dict of key-value pairs to match against data fields
            limit: Max results to return

        Returns:
            List of matching data dicts, sorted by most recent first
        """
        cat_dir = os.path.join(self.base_dir, category)
        if not os.path.exists(cat_dir):
            return []

        results = []
        for filename in os.listdir(cat_dir):
            if not filename.endswith('.json'):
                continue
            try:
                filepath = os.path.join(cat_dir, filename)
                with open(filepath, 'r') as f:
                    entry = json.load(f)

                data = entry.get('data', {})

                # Apply filters
                if filters:
                    match = all(
                        data.get(k) == v for k, v in filters.items()
                    )
                    if not match:
                        continue

                results.append({
                    'key': entry.get('key'),
                    'timestamp': entry.get('timestamp', 0),
                    'data': data,
                })
            except Exception:
                continue

        # Sort by most recent
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return results[:limit]

    def store_session(self, session_id: str, shots: list, conditions: dict,
                      feedback: str = None):
        """
        Convenience: store a complete flight session.
        """
        self.store('sessions', session_id, {
            'shots': shots,
            'conditions': conditions,
            'feedback': feedback,
            'shot_count': len(shots),
        })

    def store_user_preference(self, user_id: str, preferences: dict):
        """
        Convenience: store/update user preferences.
        Merges with existing preferences.
        """
        existing = self.recall('preferences', user_id) or {}
        existing.update(preferences)
        self.store('preferences', user_id, existing)

    def get_user_preferences(self, user_id: str) -> dict:
        """Get user preferences, with defaults."""
        defaults = {
            'preferred_shot_style': 'cinematic',
            'color_profile': 'natural',
            'max_speed': 3.0,
            'default_altitude': 5.0,
            'hypersmooth': 'auto',
        }
        prefs = self.recall('preferences', user_id) or {}
        defaults.update(prefs)
        return defaults

    def store_location(self, location_id: str, data: dict):
        """Store location-specific knowledge (obstacles, good angles, etc)."""
        self.store('locations', location_id, data)

    def get_location_knowledge(self, lat: float, lon: float, radius_m: float = 100) -> list:
        """
        Find stored knowledge about locations near the given GPS coordinates.
        Simple distance check against all stored locations.
        """
        import math
        results = []
        locations = self.search('locations', limit=100)

        for loc in locations:
            data = loc.get('data', {})
            loc_lat = data.get('lat', 0)
            loc_lon = data.get('lon', 0)

            # Approximate distance
            dlat = (lat - loc_lat) * 111320
            dlon = (lon - loc_lon) * 111320 * math.cos(math.radians(lat))
            dist = math.sqrt(dlat**2 + dlon**2)

            if dist <= radius_m:
                results.append({**data, '_distance_m': round(dist, 1)})

        return results

    def store_pattern(self, pattern_name: str, shot_sequence: list, context: dict):
        """
        Store a successful cinematic pattern (sequence of shots that worked well).
        """
        self.store('patterns', pattern_name, {
            'sequence': shot_sequence,
            'context': context,
        })

    def find_patterns(self, context: dict) -> list:
        """Find cinematic patterns that match the current context."""
        return self.search('patterns', filters=context, limit=5)

    def delete(self, category: str, key: str) -> bool:
        """Delete a memory entry."""
        filepath = os.path.join(self.base_dir, category, f"{key}.json")
        try:
            os.remove(filepath)
            logger.info(f"Memory deleted: {category}/{key}")
            return True
        except FileNotFoundError:
            return False

    def get_stats(self) -> dict:
        """Get memory storage statistics."""
        stats = {}
        for cat in self.categories:
            cat_dir = os.path.join(self.base_dir, cat)
            if os.path.exists(cat_dir):
                files = [f for f in os.listdir(cat_dir) if f.endswith('.json')]
                stats[cat] = len(files)
            else:
                stats[cat] = 0
        return stats
