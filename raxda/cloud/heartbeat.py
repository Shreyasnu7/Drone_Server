"""
raxda/cloud/heartbeat.py — Cloud heartbeat sender for the Radxa onboard agent.

Sends periodic heartbeat to the cloud server so it knows the drone is alive.
Also receives configuration updates and emergency commands from cloud.

Data Flow:
    Radxa ──heartbeat (5s interval)──▶ Cloud Server
    Radxa ◀──config/emergency────────── Cloud Server
"""
import time
import json
import logging
import threading

logger = logging.getLogger('heartbeat')

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class CloudHeartbeat:
    """
    Sends periodic heartbeat to cloud and checks for emergency commands.

    Usage:
        hb = CloudHeartbeat(drone_id='DRONE001', server_url='https://...')
        hb.start()

        # In your main loop:
        hb.update_telemetry(telemetry_dict)

        # Check for cloud commands:
        cmd = hb.get_pending_command()
    """

    def __init__(self, drone_id, server_url, interval=5.0):
        """
        Args:
            drone_id: Unique drone identifier (e.g., 'RADXA_X', 'DRONE001')
            server_url: Cloud server base URL (e.g., 'https://drone-server-r0qe.onrender.com')
            interval: Heartbeat interval in seconds
        """
        self.drone_id = drone_id
        self.server_url = server_url.rstrip('/')
        self.interval = interval
        self.running = False
        self._thread = None
        self._telemetry = {}
        self._pending_commands = []
        self._lock = threading.Lock()
        self.last_success = 0
        self.consecutive_failures = 0

    def start(self):
        """Start heartbeat thread."""
        if not REQUESTS_AVAILABLE:
            logger.error("Cannot start heartbeat: 'requests' not installed")
            return False

        self.running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info(f"Heartbeat started: {self.drone_id} → {self.server_url}")
        return True

    def stop(self):
        """Stop heartbeat thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def update_telemetry(self, telemetry):
        """Update telemetry data to include in next heartbeat."""
        with self._lock:
            self._telemetry = dict(telemetry)

    def get_pending_command(self):
        """Get next pending command from cloud, or None."""
        with self._lock:
            if self._pending_commands:
                return self._pending_commands.pop(0)
        return None

    @property
    def is_connected(self):
        """True if last heartbeat succeeded within 2x interval."""
        return (time.time() - self.last_success) < (self.interval * 2)

    def _heartbeat_loop(self):
        """Background thread: send heartbeats and check for commands."""
        while self.running:
            try:
                self._send_heartbeat()
                self._check_commands()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self.consecutive_failures += 1

            time.sleep(self.interval)

    def _send_heartbeat(self):
        """Send heartbeat with current telemetry to cloud."""
        with self._lock:
            telem = dict(self._telemetry)

        payload = {
            'drone_id': self.drone_id,
            'timestamp': time.time(),
            'status': 'online',
            'telemetry': {
                'battery': telem.get('battery', telem.get('battery_pct', -1)),
                'altitude': telem.get('alt_rel', telem.get('altitude', 0)),
                'lat': telem.get('lat', 0),
                'lon': telem.get('lon', 0),
                'armed': telem.get('armed', False),
                'mode': telem.get('mode', 'UNKNOWN'),
                'satellites': telem.get('satellites', 0),
                'ground_speed': telem.get('ground_speed', 0),
            },
        }

        try:
            url = f"{self.server_url}/api/heartbeat"
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                self.last_success = time.time()
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures % 5 == 0:
                    logger.warning(
                        f"Heartbeat failed: HTTP {resp.status_code} "
                        f"(x{self.consecutive_failures})"
                    )
        except requests.exceptions.Timeout:
            self.consecutive_failures += 1
        except requests.exceptions.ConnectionError:
            self.consecutive_failures += 1
            if self.consecutive_failures % 10 == 0:
                logger.warning(f"Cloud unreachable (x{self.consecutive_failures})")

    def _check_commands(self):
        """Poll cloud for pending commands for this drone."""
        try:
            url = f"{self.server_url}/api/command/{self.drone_id}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, dict) and data.get('action'):
                    with self._lock:
                        self._pending_commands.append(data)
                    logger.info(f"Cloud command received: {data.get('action')}")
        except Exception:
            pass  # Non-critical — commands also come via WebSocket
