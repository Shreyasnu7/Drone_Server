"""
raxda/cloud/auth.py — Authentication for drone-to-cloud communication.

Handles:
- Token generation and validation for WebSocket connections
- Drone identity verification
- API key management for cloud endpoints

Used by:
- raxda/main.py: Authenticates before connecting to cloud
- raxda_bridge/real_bridge_service.py: Token in WS handshake
"""
import hashlib
import time
import os
import json
import logging

logger = logging.getLogger('auth')


class DroneAuth:
    """
    Simple token-based authentication for drone-to-cloud.

    The drone generates a bearer token from its DRONE_ID + shared secret.
    The cloud server validates this token before accepting commands.

    For production: replace with proper JWT or mTLS.
    """

    def __init__(self, drone_id, secret=None):
        """
        Args:
            drone_id: Unique drone identifier (e.g., 'RADXA_X')
            secret: Shared secret for token generation.
                    Falls back to DRONE_SECRET env var, then a default.
        """
        self.drone_id = drone_id
        self.secret = secret or os.environ.get('DRONE_SECRET', 'antigravity_drone_2024')
        self._token = None
        self._token_expiry = 0

    def generate_token(self, ttl=3600):
        """
        Generate a bearer token valid for ttl seconds.

        Token format: sha256(drone_id + secret + timestamp_hour)
        This ensures tokens rotate hourly but are deterministic
        so both drone and server can compute the same token.
        """
        # Round timestamp to the hour for deterministic token
        hour_ts = str(int(time.time()) // 3600)
        raw = f"{self.drone_id}:{self.secret}:{hour_ts}"
        token = hashlib.sha256(raw.encode()).hexdigest()
        self._token = token
        self._token_expiry = time.time() + ttl
        return token

    def get_token(self):
        """Get current valid token, regenerating if expired."""
        if self._token is None or time.time() > self._token_expiry:
            self.generate_token()
        return self._token

    def get_auth_header(self):
        """Get HTTP Authorization header value."""
        return f"Bearer {self.get_token()}"

    def get_ws_handshake(self):
        """
        Get WebSocket handshake payload.
        Sent as the first message after WS connection.
        """
        return json.dumps({
            'type': 'connect_drone',
            'droneId': self.drone_id,
            'token': self.get_token(),
            'timestamp': time.time(),
        })

    @staticmethod
    def validate_token(drone_id, token, secret=None):
        """
        Server-side: validate a token from a drone.
        Checks current hour and previous hour (to handle clock skew).
        """
        secret = secret or os.environ.get('DRONE_SECRET', 'antigravity_drone_2024')
        now_hour = int(time.time()) // 3600

        for offset in [0, -1]:  # Check current and previous hour
            raw = f"{drone_id}:{secret}:{now_hour + offset}"
            expected = hashlib.sha256(raw.encode()).hexdigest()
            if token == expected:
                return True

        return False
