"""
cloud_ai/dispatcher.py — Dispatches AI plans to downstream systems.

Routes orchestrator decisions to:
1. Laptop AI (Camera Brain) — cinematic plan, gimbal, camera settings
2. Radxa Bridge (Drone) — flight commands, safety envelope, waypoints

Uses the WebSocket connections managed by ws_router.py's ConnectionManager.
The dispatcher holds references to active WebSocket connections and sends
JSON payloads directly.
"""
import json
import logging
import asyncio

logger = logging.getLogger('dispatcher')


class OrchestrationDispatcher:
    """
    Dispatches plans to downstream systems via WebSocket.

    Lifecycle:
        1. ws_router.py calls register_drone_connection(ws) when drone connects
        2. Orchestrator calls dispatch_to_laptop/dispatch_to_radxa with plan data
        3. Dispatcher serializes and sends via the active WebSocket
    """

    def __init__(self):
        self._drone_ws = None       # WebSocket to Radxa bridge
        self._laptop_ws = None      # WebSocket to laptop AI (same as drone for now)
        self._mobile_clients = []   # WebSocket list to mobile apps

    def register_drone_connection(self, ws):
        """Called by ws_router when a drone/laptop connects."""
        self._drone_ws = ws
        self._laptop_ws = ws  # Laptop AI connects via same WS as drone
        logger.info("Dispatcher: Drone/Laptop WebSocket registered")

    def register_mobile_connection(self, ws):
        """Called by ws_router when a mobile app connects."""
        self._mobile_clients.append(ws)

    def remove_drone_connection(self):
        """Called by ws_router when drone disconnects."""
        self._drone_ws = None
        self._laptop_ws = None
        logger.info("Dispatcher: Drone/Laptop WebSocket removed")

    def remove_mobile_connection(self, ws):
        if ws in self._mobile_clients:
            self._mobile_clients.remove(ws)

    async def dispatch_to_laptop(self, plan: dict):
        """
        Sends cinematic plan to Laptop AI (Camera Brain).

        Plan structure expected by director_core.py:
        {
            "type": "ai_plan",
            "payload": {
                "shot_type": "orbit" | "follow" | "dolly" | "crane" | "hover",
                "motion_energy": 0.0-1.0,
                "camera": {
                    "exposure_compensation": int,
                    "white_balance": str,
                    "color_profile": str,
                    "hypersmooth": str,
                },
                "gimbal": {
                    "pitch": float,
                    "yaw": float,
                },
                "waypoints": [...],
            }
        }
        """
        if self._laptop_ws is None:
            logger.warning("Dispatch to laptop failed: no connection")
            return False

        try:
            message = json.dumps({
                "type": "ai_plan",
                "payload": plan,
            })
            await self._laptop_ws.send_text(message)
            logger.info(f"Dispatched plan to laptop: {plan.get('shot_type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Dispatch to laptop failed: {e}")
            self._laptop_ws = None
            return False

    async def dispatch_to_radxa(self, safety_constraints: dict):
        """
        Sends safety envelope and flight commands to Radxa bridge.

        Structure expected by real_bridge_service.py:
        {
            "type": "ai_safety",
            "payload": {
                "max_speed": float,
                "max_altitude": float,
                "geofence": {"lat": float, "lon": float, "radius_m": float},
                "no_fly_zones": [...],
                "min_battery_rtl": int,
            }
        }
        """
        if self._drone_ws is None:
            logger.warning("Dispatch to radxa failed: no connection")
            return False

        try:
            message = json.dumps({
                "type": "ai_safety",
                "payload": safety_constraints,
            })
            await self._drone_ws.send_text(message)
            logger.info("Dispatched safety envelope to Radxa")
            return True
        except Exception as e:
            logger.error(f"Dispatch to radxa failed: {e}")
            self._drone_ws = None
            return False

    async def dispatch_flight_command(self, command: str, payload: dict = None):
        """
        Send a direct flight command to the drone (ARM, TAKEOFF, LAND, etc).
        This is relayed by the Radxa bridge to the flight controller.
        """
        if self._drone_ws is None:
            logger.warning(f"Dispatch command '{command}' failed: no drone connection")
            return False

        try:
            message = json.dumps({
                "type": command.upper(),
                "payload": payload or {},
            })
            await self._drone_ws.send_text(message)
            logger.info(f"Dispatched flight command: {command}")
            return True
        except Exception as e:
            logger.error(f"Dispatch flight command failed: {e}")
            return False

    async def dispatch_gopro_settings(self, settings: dict):
        """
        Send GoPro camera settings to the drone's GoPro proxy.
        Handled by real_bridge_service.py → gopro_proxy.py → GoPro BLE.
        """
        if self._drone_ws is None:
            logger.warning("Dispatch GoPro settings failed: no connection")
            return False

        try:
            message = json.dumps({
                "type": "GOPRO_SETTINGS",
                "payload": {"settings": settings},
            })
            await self._drone_ws.send_text(message)
            logger.info(f"Dispatched GoPro settings: {list(settings.keys())}")
            return True
        except Exception as e:
            logger.error(f"Dispatch GoPro settings failed: {e}")
            return False

    async def broadcast_to_mobile(self, event_type: str, data: dict):
        """Send a status update to all connected mobile apps."""
        dead = []
        message = json.dumps({"type": event_type, "payload": data})

        for ws in self._mobile_clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._mobile_clients.remove(ws)

    @property
    def is_drone_connected(self):
        return self._drone_ws is not None

    @property
    def is_laptop_connected(self):
        return self._laptop_ws is not None