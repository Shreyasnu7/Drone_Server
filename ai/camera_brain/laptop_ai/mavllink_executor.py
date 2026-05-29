"""
MavlinkExecutor: Dynamic MAVLink command dispatcher.

Translates high-level AI commands into MAVLink protocol commands.
Does NOT restrict what the AI can request — it dynamically maps
whatever the AI sends to the closest MAVLink equivalent.
"""

import logging
import time

logger = logging.getLogger(__name__)


class MavlinkExecutor:
    """
    Dynamic command executor for ArduPilot via MAVLink.
    
    The AI sends free-form commands. This module translates them
    into MAVLink protocol calls. If a command isn't recognized,
    it attempts to interpret it rather than dropping it.
    """

    def __init__(self, bridge=None, autopilot=None):
        self.bridge = bridge
        self.autopilot = autopilot
        self.last_cmd_time = 0
        self.cmd_count = 0
        print("✅ MavlinkExecutor initialized")

    def execute(self, cmd: dict):
        """
        Execute ANY command dict from the AI.
        
        Tries to intelligently map whatever the AI sends to
        actual MAVLink calls. No fixed action list — reads the 
        command dynamically and extracts usable parameters.
        """
        self.last_cmd_time = time.time()
        self.cmd_count += 1
        
        action = str(cmd.get("action", cmd.get("cmd", ""))).upper().strip()
        logger.info(f"🎯 MavlinkExecutor #{self.cmd_count}: {action} | {cmd}")

        if not self.autopilot and not self.bridge:
            logger.warning("No autopilot or bridge — cannot execute")
            return False

        # If no direct autopilot, relay everything through bridge
        if not self.autopilot:
            return self._relay_to_bridge(cmd)

        try:
            return self._dispatch(action, cmd)
        except Exception as e:
            logger.error(f"MavlinkExecutor error: {e}")
            return False

    def _dispatch(self, action: str, cmd: dict):
        """
        Dynamically dispatch based on action string.
        Handles exact matches AND fuzzy/partial matches.
        """
        # === VELOCITY / MOVEMENT ===
        if any(k in action for k in ["VELOCITY", "MOVE", "FLY", "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STRAFE"]):
            vx = cmd.get("vx", cmd.get("forward", cmd.get("speed", 0)))
            vy = cmd.get("vy", cmd.get("lateral", 0))
            vz = cmd.get("vz", cmd.get("vertical", 0))
            
            # Handle directional keywords in action
            if "FORWARD" in action:
                vx = abs(vx) if vx else cmd.get("value", 1.0)
            elif "BACKWARD" in action or "BACK" in action:
                vx = -(abs(vx) if vx else cmd.get("value", 1.0))
            elif "LEFT" in action:
                vy = -(abs(vy) if vy else cmd.get("value", 1.0))
            elif "RIGHT" in action:
                vy = abs(vy) if vy else cmd.get("value", 1.0)
            
            self.autopilot.send_velocity(float(vx), float(vy), float(vz))
            return True

        # === ALTITUDE ===
        elif any(k in action for k in ["ASCEND", "DESCEND", "CLIMB", "DROP", "ALTITUDE"]):
            rate = cmd.get("rate", cmd.get("value", cmd.get("speed", 1.0)))
            if any(k in action for k in ["DESCEND", "DROP"]):
                rate = abs(rate)  # Positive vz = descend in NED
            else:
                rate = -abs(rate)  # Negative vz = ascend in NED
            self.autopilot.send_velocity(0, 0, float(rate))
            return True

        # === TAKEOFF ===
        elif "TAKEOFF" in action or "TAKE_OFF" in action or "LAUNCH" in action:
            alt = cmd.get("altitude", cmd.get("alt", cmd.get("height", cmd.get("value", 3.0))))
            self.autopilot.takeoff(float(alt))
            return True

        # === LAND ===
        elif "LAND" in action or "TOUCH_DOWN" in action:
            self.autopilot.land()
            return True

        # === GOTO / WAYPOINT ===
        elif any(k in action for k in ["GOTO", "GO_TO", "WAYPOINT", "NAVIGATE", "FLY_TO"]):
            lat = cmd.get("lat", cmd.get("latitude"))
            lng = cmd.get("lng", cmd.get("lon", cmd.get("longitude")))
            alt = cmd.get("alt", cmd.get("altitude", 10))
            if lat and lng:
                self.autopilot.goto(float(lat), float(lng), float(alt))
                return True
            logger.warning("GOTO needs lat/lng")
            return False

        # === YAW / HEADING / ROTATE / SPIN ===
        elif any(k in action for k in ["YAW", "HEADING", "ROTATE", "SPIN", "TURN"]):
            heading = cmd.get("heading", cmd.get("angle", cmd.get("degrees", cmd.get("value", 0))))
            rate = cmd.get("rate", cmd.get("speed", 45))
            self.autopilot.set_yaw(float(heading), float(rate))
            return True

        # === MODE CHANGE ===
        elif any(k in action for k in ["MODE", "SET_MODE", "GUIDED", "LOITER", "RTL", "STABILIZE", "AUTO"]):
            # If the action IS the mode name
            mode = cmd.get("mode", action)
            for m in ["GUIDED", "LOITER", "RTL", "STABILIZE", "AUTO", "LAND", "ALT_HOLD", "POSHOLD"]:
                if m in action:
                    mode = m
                    break
            self.autopilot.set_mode(mode)
            return True

        # === RETURN HOME ===
        elif any(k in action for k in ["RTH", "RETURN", "HOME", "RTL"]):
            self.autopilot.set_mode("RTL")
            return True

        # === STOP / HOVER / BRAKE / HOLD ===
        elif any(k in action for k in ["STOP", "HOVER", "BRAKE", "HOLD", "PAUSE", "FREEZE"]):
            self.autopilot.send_velocity(0, 0, 0)
            return True

        # === RC OVERRIDE (raw channel control) ===
        elif "RC" in action or "OVERRIDE" in action or "CHANNEL" in action:
            channels = cmd.get("channels", [1500, 1500, 1500, 1500])
            self.autopilot.rc_override(channels)
            return True

        # === ORBIT / CIRCLE ===
        elif any(k in action for k in ["ORBIT", "CIRCLE", "SPIRAL"]):
            # Orbit requires special handling — relay to bridge for the AI planner
            return self._relay_to_bridge(cmd)

        # === FLIP / ACROBATICS ===
        elif any(k in action for k in ["FLIP", "ROLL_OVER", "ACRO", "BARREL"]):
            return self._relay_to_bridge(cmd)

        # === UNKNOWN — try to relay to bridge for AI handling ===
        else:
            logger.warning(f"Unknown action '{action}' — relaying to bridge for AI interpretation")
            return self._relay_to_bridge(cmd)

    def _relay_to_bridge(self, cmd: dict):
        """Send command through WebSocket bridge for remote handling."""
        if self.bridge:
            try:
                self.bridge.send_message({"type": "command", "payload": cmd})
                return True
            except Exception as e:
                logger.error(f"Bridge relay failed: {e}")
        return False

    def emergency_stop(self):
        """Immediate all-axis stop."""
        logger.warning("🛑 EMERGENCY STOP")
        if self.autopilot:
            self.autopilot.send_velocity(0, 0, 0)
        return True
