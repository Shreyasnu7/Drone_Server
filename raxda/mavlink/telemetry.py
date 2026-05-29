"""
raxda/mavlink/telemetry.py — Lightweight telemetry reader for the Radxa onboard agent.

Reads MAVLink messages from the flight controller and provides a simple
dict-based API for the decision engine to query current state.

Unlike mavlink/telemetry.py (laptop-side, full featured), this is optimized
for the Radxa Zero 3W — minimal memory, no dependencies beyond pymavlink.

Data Flow:
    FC (ArduPilot) --UART--> RaxdaTelemetry.update() --> decision.py
"""
import time
import math


class RaxdaTelemetry:
    """
    Lightweight telemetry receiver for onboard use.

    Usage:
        from pymavlink import mavutil
        master = mavutil.mavlink_connection('/dev/ttyS2', baud=57600)
        telem = RaxdaTelemetry(master)

        while True:
            telem.update()  # Non-blocking, reads available messages
            state = telem.get_state()
            print(f"Alt: {state['alt_rel']}m  Armed: {state['armed']}")
    """

    def __init__(self, master):
        self.master = master
        self.state = {
            'lat': 0.0,
            'lon': 0.0,
            'alt_rel': 0.0,
            'alt_msl': 0.0,
            'heading': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'ground_speed': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'battery': -1,
            'voltage': 0.0,
            'current': 0.0,
            'satellites': 0,
            'gps_fix': 0,
            'armed': False,
            'mode': 'UNKNOWN',
            'mode_id': 0,
            'last_heartbeat': 0.0,
        }

        # ArduCopter mode map
        self._mode_map = {
            0: 'STABILIZE', 2: 'ALT_HOLD', 3: 'AUTO',
            4: 'GUIDED', 5: 'LOITER', 6: 'RTL',
            9: 'LAND', 16: 'POSHOLD',
        }

    def update(self, max_messages=20):
        """
        Read up to max_messages from FC. Non-blocking.
        Call this every loop iteration (~10-50Hz).
        """
        from pymavlink import mavutil as mu

        for _ in range(max_messages):
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break

            msg_type = msg.get_type()

            if msg_type == 'HEARTBEAT':
                self.state['last_heartbeat'] = time.time()
                self.state['armed'] = bool(
                    msg.base_mode & mu.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.state['mode_id'] = msg.custom_mode
                self.state['mode'] = self._mode_map.get(
                    msg.custom_mode, f'MODE_{msg.custom_mode}'
                )

            elif msg_type == 'GLOBAL_POSITION_INT':
                self.state['lat'] = msg.lat / 1e7
                self.state['lon'] = msg.lon / 1e7
                self.state['alt_msl'] = msg.alt / 1000.0
                self.state['alt_rel'] = msg.relative_alt / 1000.0
                self.state['vx'] = msg.vx / 100.0
                self.state['vy'] = msg.vy / 100.0
                self.state['vz'] = msg.vz / 100.0
                self.state['heading'] = msg.hdg / 100.0
                self.state['ground_speed'] = math.sqrt(
                    self.state['vx']**2 + self.state['vy']**2
                )

            elif msg_type == 'GPS_RAW_INT':
                self.state['satellites'] = msg.satellites_visible
                self.state['gps_fix'] = msg.fix_type

            elif msg_type == 'ATTITUDE':
                self.state['roll'] = msg.roll
                self.state['pitch'] = msg.pitch
                self.state['yaw'] = msg.yaw

            elif msg_type == 'SYS_STATUS':
                self.state['battery'] = msg.battery_remaining
                self.state['voltage'] = msg.voltage_battery / 1000.0
                self.state['current'] = msg.current_battery / 100.0

    def get_state(self):
        """Returns a copy of current telemetry state dict."""
        return dict(self.state)

    def is_armed(self):
        return self.state['armed']

    def get_mode(self):
        return self.state['mode']

    def is_data_fresh(self, max_age=3.0):
        """Check if we've received a heartbeat recently."""
        return (time.time() - self.state['last_heartbeat']) < max_age

    def get_position(self):
        return (self.state['lat'], self.state['lon'], self.state['alt_rel'])

    def get_velocity_ned(self):
        return (self.state['vx'], self.state['vy'], self.state['vz'])
