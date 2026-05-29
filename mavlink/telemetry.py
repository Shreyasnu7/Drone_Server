"""
mavlink/telemetry.py — Real-time telemetry receiver from Flight Controller.
Reads MAVLink messages and provides position, velocity, attitude, battery, GPS.
Used by control/move.py for closed-loop navigation.

Data Flow:
    FC (ArduPilot) --MAVLink--> TelemetryReceiver --get_*()--> control/move.py
                                                  --get_all()--> raxda_bridge (broadcast)
"""
import threading
import time
import math
from pymavlink import mavutil


class TelemetryReceiver:
    """
    Continuously reads MAVLink telemetry from the FC in a background thread.
    Thread-safe: all public methods acquire self.lock before reading state.

    Usage:
        from mavlink.connection import connect
        from mavlink.telemetry import TelemetryReceiver

        master = connect()
        telem = TelemetryReceiver(master)

        while True:
            pos = telem.get_position()
            print(f"Alt: {pos['alt_rel']}m  Heading: {pos['heading']}°")
            time.sleep(0.5)
    """

    def __init__(self, master):
        self.master = master
        self.lock = threading.Lock()
        self.running = True

        # ── Position (GPS + Baro fused) ──
        self.lat = 0.0          # degrees
        self.lon = 0.0          # degrees
        self.alt_rel = 0.0      # meters relative to home
        self.alt_msl = 0.0      # meters above sea level
        self.home_lat = None
        self.home_lon = None
        self.home_alt = None

        # ── Velocity (NED frame, m/s) ──
        self.vx = 0.0           # North
        self.vy = 0.0           # East
        self.vz = 0.0           # Down (positive = descending)
        self.ground_speed = 0.0

        # ── Attitude (radians) ──
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0          # -π..π, North=0

        # ── Battery ──
        self.battery_pct = -1   # -1 = unknown
        self.voltage = 0.0      # Volts
        self.current = 0.0      # Amps

        # ── GPS quality ──
        self.satellites = 0
        self.gps_fix = 0        # 0=no, 2=2D, 3=3D

        # ── Armed / Mode ──
        self.armed = False
        self.flight_mode = "UNKNOWN"
        self.mode_id = 0

        # ── Heading ──
        self.heading_deg = 0.0  # 0..360

        # ── Timestamps (for staleness detection) ──
        self.last_heartbeat = 0.0
        self.last_gps = 0.0
        self.last_attitude = 0.0
        self.last_battery = 0.0

        # ── Start background reader ──
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """Background thread: continuously read and parse MAVLink messages."""
        # ArduCopter mode ID → name mapping
        MODE_MAP = {
            0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO',
            4: 'GUIDED', 5: 'LOITER', 6: 'RTL', 7: 'CIRCLE',
            9: 'LAND', 11: 'DRIFT', 13: 'SPORT', 14: 'FLIP',
            15: 'AUTOTUNE', 16: 'POSHOLD', 17: 'BRAKE',
            18: 'THROW', 19: 'AVOID_ADSB', 20: 'GUIDED_NOGPS',
            21: 'SMART_RTL', 22: 'FLOWHOLD', 23: 'FOLLOW',
            24: 'ZIGZAG', 25: 'SYSTEMID', 26: 'AUTOROTATE',
        }

        while self.running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue

                msg_type = msg.get_type()

                with self.lock:
                    # ── HEARTBEAT ──
                    if msg_type == 'HEARTBEAT':
                        self.last_heartbeat = time.time()
                        self.armed = bool(
                            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                        )
                        self.mode_id = msg.custom_mode
                        self.flight_mode = MODE_MAP.get(
                            msg.custom_mode, f"MODE_{msg.custom_mode}"
                        )

                    # ── GLOBAL POSITION (fused GPS+Baro, 4-10Hz) ──
                    elif msg_type == 'GLOBAL_POSITION_INT':
                        self.lat = msg.lat / 1e7
                        self.lon = msg.lon / 1e7
                        self.alt_msl = msg.alt / 1000.0
                        self.alt_rel = msg.relative_alt / 1000.0
                        self.vx = msg.vx / 100.0   # cm/s → m/s
                        self.vy = msg.vy / 100.0
                        self.vz = msg.vz / 100.0
                        self.heading_deg = msg.hdg / 100.0
                        self.ground_speed = math.sqrt(self.vx**2 + self.vy**2)
                        self.last_gps = time.time()

                    # ── RAW GPS (satellite info) ──
                    elif msg_type == 'GPS_RAW_INT':
                        self.satellites = msg.satellites_visible
                        self.gps_fix = msg.fix_type
                        # Record home on first 3D fix
                        if self.home_lat is None and msg.fix_type >= 3:
                            self.home_lat = msg.lat / 1e7
                            self.home_lon = msg.lon / 1e7
                            self.home_alt = msg.alt / 1000.0

                    # ── ATTITUDE (50Hz typically) ──
                    elif msg_type == 'ATTITUDE':
                        self.roll = msg.roll
                        self.pitch = msg.pitch
                        self.yaw = msg.yaw
                        self.last_attitude = time.time()

                    # ── BATTERY/SYSTEM STATUS ──
                    elif msg_type == 'SYS_STATUS':
                        self.battery_pct = msg.battery_remaining
                        self.voltage = msg.voltage_battery / 1000.0
                        self.current = msg.current_battery / 100.0
                        self.last_battery = time.time()

                    # ── HOME POSITION (set on arm) ──
                    elif msg_type == 'HOME_POSITION':
                        self.home_lat = msg.latitude / 1e7
                        self.home_lon = msg.longitude / 1e7
                        self.home_alt = msg.altitude / 1000.0

            except Exception:
                time.sleep(0.01)

    # ═══════════════════════════════════════════
    # PUBLIC API — all thread-safe
    # ═══════════════════════════════════════════

    def get_position(self):
        """Returns current GPS position + relative altitude."""
        with self.lock:
            return {
                'lat': self.lat,
                'lon': self.lon,
                'alt_rel': self.alt_rel,
                'alt_msl': self.alt_msl,
                'heading': self.heading_deg,
            }

    def get_velocity(self):
        """Returns NED velocity in m/s."""
        with self.lock:
            return {
                'vx': self.vx,  # North
                'vy': self.vy,  # East
                'vz': self.vz,  # Down
                'ground_speed': self.ground_speed,
            }

    def get_attitude(self):
        """Returns roll, pitch, yaw in radians."""
        with self.lock:
            return {
                'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw,
            }

    def get_battery(self):
        """Returns battery percentage, voltage, current."""
        with self.lock:
            return {
                'pct': self.battery_pct,
                'voltage': self.voltage,
                'current': self.current,
            }

    def get_gps(self):
        """Returns GPS fix quality and home position."""
        with self.lock:
            return {
                'fix': self.gps_fix,
                'satellites': self.satellites,
                'home': (self.home_lat, self.home_lon, self.home_alt),
            }

    def is_armed(self):
        with self.lock:
            return self.armed

    def get_mode(self):
        with self.lock:
            return self.flight_mode

    def is_data_fresh(self, max_age=3.0):
        """Check if telemetry data is recent enough to trust."""
        now = time.time()
        with self.lock:
            return (now - self.last_heartbeat) < max_age

    def get_all(self):
        """Full telemetry snapshot — used by bridge for broadcast."""
        with self.lock:
            return {
                'lat': self.lat, 'lon': self.lon,
                'alt_rel': self.alt_rel, 'alt_msl': self.alt_msl,
                'heading': self.heading_deg,
                'vx': self.vx, 'vy': self.vy, 'vz': self.vz,
                'ground_speed': self.ground_speed,
                'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw,
                'battery_pct': self.battery_pct,
                'voltage': self.voltage, 'current': self.current,
                'satellites': self.satellites, 'gps_fix': self.gps_fix,
                'armed': self.armed, 'mode': self.flight_mode,
                'mode_id': self.mode_id,
                'home': (self.home_lat, self.home_lon, self.home_alt),
                'last_heartbeat': self.last_heartbeat,
            }

    def stop(self):
        """Stop the background telemetry thread."""
        self.running = False
        self._thread.join(timeout=2)
