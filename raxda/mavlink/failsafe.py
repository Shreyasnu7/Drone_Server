"""
raxda/mavlink/failsafe.py — Failsafe logic for the onboard drone agent.

Monitors critical conditions and triggers emergency responses:
- Battery critically low → auto-land or RTL
- Lost cloud connection → hold position, then RTL
- Lost FC heartbeat → emergency disarm (crash prevention)
- Altitude violation → force descend
- Geofence breach → RTL
- Excessive tilt → emergency land (flip/tumble prevention)

This runs ONBOARD the drone on the Radxa Zero 3W.
It operates independently of cloud/laptop — last line of defense.
"""
import time
import math
import logging

logger = logging.getLogger('failsafe')


class FailsafeManager:
    """
    Onboard failsafe monitor. Call check() every tick (~10Hz).

    Usage:
        failsafe = FailsafeManager(master)
        while True:
            action = failsafe.check(telemetry, cloud_connected=True)
            if action:
                execute_failsafe(action)
    """

    # Thresholds
    BATT_WARN = 20          # Percent — warn the user
    BATT_CRITICAL = 10      # Percent — force RTL
    BATT_EMERGENCY = 5      # Percent — force LAND immediately

    MAX_ALTITUDE = 120      # meters — absolute ceiling
    MAX_TILT_DEG = 60       # degrees — excessive tilt = tumble risk

    CLOUD_TIMEOUT = 30      # seconds — lost connection threshold
    FC_TIMEOUT = 3          # seconds — no heartbeat from FC

    GEOFENCE_RADIUS = 500   # meters from home — max distance

    def __init__(self, master):
        """
        Args:
            master: pymavlink connection to flight controller
        """
        self.master = master
        self.last_cloud_contact = time.time()
        self.last_fc_heartbeat = time.time()
        self.home_lat = None
        self.home_lon = None
        self.triggered = set()  # Track which failsafes have been triggered
        self._last_log = 0

    def set_home(self, lat, lon):
        """Set home position for geofence check."""
        self.home_lat = lat
        self.home_lon = lon
        logger.info(f"Failsafe home set: ({lat:.6f}, {lon:.6f})")

    def cloud_heartbeat(self):
        """Call this when a cloud message is received."""
        self.last_cloud_contact = time.time()

    def fc_heartbeat(self):
        """Call this when an FC heartbeat is received."""
        self.last_fc_heartbeat = time.time()

    def check(self, telemetry, cloud_connected=True):
        """
        Run all failsafe checks. Call every tick.

        Args:
            telemetry: dict with at least {battery, alt_rel, roll, pitch, lat, lon, armed}
            cloud_connected: bool — whether cloud WebSocket is active

        Returns:
            None if everything is OK, or a dict:
            {
                'action': 'LAND' | 'RTL' | 'DISARM' | 'HOLD' | 'DESCEND',
                'reason': str,
                'severity': 'warn' | 'critical' | 'emergency',
            }
        """
        now = time.time()
        armed = telemetry.get('armed', False)

        if not armed:
            # Reset triggered failsafes when disarmed
            self.triggered.clear()
            return None

        battery = telemetry.get('battery', telemetry.get('battery_pct', -1))
        alt = telemetry.get('alt_rel', telemetry.get('altitude', 0))
        roll = telemetry.get('roll', 0)
        pitch = telemetry.get('pitch', 0)
        lat = telemetry.get('lat', 0)
        lon = telemetry.get('lon', 0)

        # ── 1. Battery Failsafe ──
        if battery >= 0:
            if battery < self.BATT_EMERGENCY and 'batt_emergency' not in self.triggered:
                self.triggered.add('batt_emergency')
                logger.critical(f"BATTERY EMERGENCY: {battery}% — LANDING NOW")
                self._send_land()
                return {
                    'action': 'LAND',
                    'reason': f'Battery emergency: {battery}%',
                    'severity': 'emergency',
                }

            if battery < self.BATT_CRITICAL and 'batt_critical' not in self.triggered:
                self.triggered.add('batt_critical')
                logger.warning(f"BATTERY CRITICAL: {battery}% — RTL")
                self._send_rtl()
                return {
                    'action': 'RTL',
                    'reason': f'Battery critical: {battery}%',
                    'severity': 'critical',
                }

            if battery < self.BATT_WARN and 'batt_warn' not in self.triggered:
                self.triggered.add('batt_warn')
                logger.warning(f"BATTERY WARNING: {battery}%")
                return {
                    'action': 'HOLD',
                    'reason': f'Battery low: {battery}%',
                    'severity': 'warn',
                }

        # ── 2. Altitude Ceiling ──
        if alt > self.MAX_ALTITUDE:
            logger.warning(f"ALTITUDE VIOLATION: {alt:.1f}m > {self.MAX_ALTITUDE}m")
            return {
                'action': 'DESCEND',
                'reason': f'Altitude ceiling: {alt:.1f}m',
                'severity': 'critical',
            }

        # ── 3. Excessive Tilt (Tumble Detection) ──
        roll_deg = abs(math.degrees(roll))
        pitch_deg = abs(math.degrees(pitch))
        if roll_deg > self.MAX_TILT_DEG or pitch_deg > self.MAX_TILT_DEG:
            logger.critical(f"EXCESSIVE TILT: roll={roll_deg:.0f}° pitch={pitch_deg:.0f}°")
            # Emergency disarm — drone may be tumbling
            if roll_deg > 80 or pitch_deg > 80:
                self._send_disarm_force()
                return {
                    'action': 'DISARM',
                    'reason': f'Tumble detected: roll={roll_deg:.0f}° pitch={pitch_deg:.0f}°',
                    'severity': 'emergency',
                }
            return {
                'action': 'LAND',
                'reason': f'Excessive tilt: roll={roll_deg:.0f}° pitch={pitch_deg:.0f}°',
                'severity': 'critical',
            }

        # ── 4. Cloud Connection Lost ──
        time_since_cloud = now - self.last_cloud_contact
        if not cloud_connected or time_since_cloud > self.CLOUD_TIMEOUT:
            if 'cloud_lost' not in self.triggered:
                self.triggered.add('cloud_lost')
                logger.warning(f"CLOUD LOST: {time_since_cloud:.0f}s — holding position")
                return {
                    'action': 'HOLD',
                    'reason': f'Cloud connection lost ({time_since_cloud:.0f}s)',
                    'severity': 'warn',
                }
            # If lost for 2x timeout, RTL
            if time_since_cloud > self.CLOUD_TIMEOUT * 2 and 'cloud_rtl' not in self.triggered:
                self.triggered.add('cloud_rtl')
                logger.warning(f"CLOUD LOST EXTENDED: {time_since_cloud:.0f}s — RTL")
                self._send_rtl()
                return {
                    'action': 'RTL',
                    'reason': f'Cloud lost extended ({time_since_cloud:.0f}s)',
                    'severity': 'critical',
                }

        # ── 5. FC Heartbeat Lost ──
        time_since_fc = now - self.last_fc_heartbeat
        if time_since_fc > self.FC_TIMEOUT:
            logger.critical(f"FC HEARTBEAT LOST: {time_since_fc:.0f}s")
            return {
                'action': 'DISARM',
                'reason': f'FC heartbeat lost ({time_since_fc:.0f}s)',
                'severity': 'emergency',
            }

        # ── 6. Geofence ──
        if self.home_lat is not None and lat != 0 and lon != 0:
            dist = self._distance_m(lat, lon, self.home_lat, self.home_lon)
            if dist > self.GEOFENCE_RADIUS and 'geofence' not in self.triggered:
                self.triggered.add('geofence')
                logger.warning(f"GEOFENCE BREACH: {dist:.0f}m from home")
                self._send_rtl()
                return {
                    'action': 'RTL',
                    'reason': f'Geofence: {dist:.0f}m from home (max {self.GEOFENCE_RADIUS}m)',
                    'severity': 'critical',
                }

        return None  # All clear

    def _send_land(self):
        """Send LAND command to FC."""
        from pymavlink import mavutil
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0, 0, 0, 0, 0, 0, 0, 0
            )
        except Exception as e:
            logger.error(f"Failed to send LAND: {e}")

    def _send_rtl(self):
        """Send RTL mode command."""
        from pymavlink import mavutil
        try:
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                6  # RTL
            )
        except Exception as e:
            logger.error(f"Failed to send RTL: {e}")

    def _send_disarm_force(self):
        """Force disarm — emergency only."""
        from pymavlink import mavutil
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 21196, 0, 0, 0, 0, 0  # 21196 = force disarm
            )
        except Exception as e:
            logger.error(f"Failed to force disarm: {e}")

    @staticmethod
    def _distance_m(lat1, lon1, lat2, lon2):
        """Quick distance in meters between two GPS points."""
        dlat = (lat2 - lat1) * 111320
        dlon = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))
        return math.sqrt(dlat**2 + dlon**2)
