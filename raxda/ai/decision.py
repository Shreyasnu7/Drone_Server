"""
raxda/ai/decision.py — Onboard AI decision engine for Radxa Zero 3W.

This is the BRAIN running on the drone itself. It fuses:
- Vision data (from raxda/ai/vision.py)
- Obstacle data (from raxda/ai/avoidance.py)
- Sensor telemetry (ToF, IMU, GPS, battery from ESP32 + FC)
- Cloud commands (from cloud server via WebSocket)

And produces:
- MAVLink velocity/position commands
- Gimbal commands (pitch/yaw to ESP32)
- LED mode commands (flight state visualization)
- Telemetry reports back to cloud

This runs when the laptop AI is disconnected or for safety-critical
real-time decisions that can't wait for cloud round-trip.

Data Flow:
    vision.py ──┐
    avoidance.py─┤
    ESP32 telem ─┼──▶ DecisionEngine.tick() ──▶ MAVLink commands
    cloud cmds ──┤                              ▶ ESP32 commands
    FC telemetry─┘                              ▶ status to cloud
"""
import time
import math
import logging
import json

logger = logging.getLogger('raxda.decision')


class FlightState:
    """Enum-like flight state machine."""
    IDLE = 'IDLE'
    ARMED = 'ARMED'
    TAKING_OFF = 'TAKING_OFF'
    HOVERING = 'HOVERING'
    EXECUTING_MISSION = 'EXECUTING_MISSION'
    FOLLOWING_SUBJECT = 'FOLLOWING_SUBJECT'
    ORBITING = 'ORBITING'
    RETURNING_HOME = 'RETURNING_HOME'
    LANDING = 'LANDING'
    EMERGENCY = 'EMERGENCY'
    AVOIDING_OBSTACLE = 'AVOIDING_OBSTACLE'


class DecisionEngine:
    """
    Onboard decision engine — runs at ~10Hz on the Radxa Zero 3W.

    Usage:
        from raxda.ai.decision import DecisionEngine
        from raxda.ai.avoidance import ObstacleAvoidance
        from raxda.ai.vision import VisionProcessor

        avoidance = ObstacleAvoidance()
        vision = VisionProcessor()
        engine = DecisionEngine(avoidance, vision)

        while running:
            actions = engine.tick(telemetry, cloud_command)
            # actions = {
            #     'velocity': (vx, vy, vz),
            #     'gimbal': (pitch, yaw),
            #     'led_mode': 16,
            #     'report': {...},
            # }
    """

    # Follow parameters
    FOLLOW_SPEED = 2.0          # m/s max follow speed
    FOLLOW_DISTANCE = 5.0       # meters behind subject
    FOLLOW_ALTITUDE = 5.0       # meters above ground

    # Orbit parameters
    ORBIT_SPEED = 1.5           # m/s tangential speed
    ORBIT_RADIUS = 8.0          # meters from center

    # Safety limits
    MAX_ALTITUDE = 50.0         # meters
    MIN_ALTITUDE = 1.5          # meters (don't go below this)
    MAX_SPEED = 5.0             # m/s
    MIN_BATTERY_PCT = 15        # auto-RTL below this

    # LED mode mapping (from esp32_driver.py constants)
    LED_MODES = {
        FlightState.IDLE: 1,             # Safe breathe
        FlightState.ARMED: 2,            # RGB breathe
        FlightState.TAKING_OFF: 10,      # Ascend
        FlightState.HOVERING: 16,        # Hover
        FlightState.EXECUTING_MISSION: 12,  # Forward
        FlightState.FOLLOWING_SUBJECT: 12,
        FlightState.ORBITING: 21,        # Audi wipe
        FlightState.RETURNING_HOME: 3,   # Police
        FlightState.LANDING: 11,         # Descend
        FlightState.EMERGENCY: 99,       # Error
        FlightState.AVOIDING_OBSTACLE: 30,  # Braking
    }

    def __init__(self, avoidance, vision=None):
        self.avoidance = avoidance
        self.vision = vision
        self.state = FlightState.IDLE
        self.prev_state = FlightState.IDLE

        # Mission state
        self.mission_waypoints = []
        self.current_waypoint_idx = 0

        # Follow state
        self.follow_target_id = None
        self.follow_lost_frames = 0

        # Orbit state
        self.orbit_center = None  # (lat, lon)
        self.orbit_angle = 0.0

        # Timing
        self.state_entered_at = time.time()
        self.last_tick = time.time()

    def _set_state(self, new_state):
        """Transition to a new flight state."""
        if new_state != self.state:
            logger.info(f"State: {self.state} → {new_state}")
            self.prev_state = self.state
            self.state = new_state
            self.state_entered_at = time.time()

    def tick(self, telemetry, cloud_command=None, tof_sensors=None, vision_result=None):
        """
        Main decision loop — call at 10-20Hz.

        Args:
            telemetry: dict from FC (alt, lat, lon, yaw, battery, armed, mode, etc)
            cloud_command: dict from cloud server or None
            tof_sensors: dict {'t1': mm, 't2': mm, ...} from ESP32
            vision_result: dict from VisionProcessor.process() or None

        Returns:
            dict of actions to execute
        """
        dt = time.time() - self.last_tick
        self.last_tick = time.time()

        actions = {
            'velocity': None,       # (vx, vy, vz) NED m/s — None means no override
            'position': None,       # (lat, lon, alt) — None means no goto
            'gimbal': None,         # (pitch, yaw) degrees — None means no change
            'led_mode': None,       # int — None means no change
            'yaw': None,            # heading degrees — None means no change
            'report': {},           # status report for cloud
            'hold': False,          # True = hold current position
            'land': False,          # True = initiate landing
            'rtl': False,           # True = return to launch
        }

        tof = tof_sensors or {}
        alt = telemetry.get('alt_rel', telemetry.get('altitude', 0))
        battery = telemetry.get('battery_pct', telemetry.get('battery', 100))
        armed = telemetry.get('armed', False)
        yaw_rad = telemetry.get('yaw', 0)

        # ── Priority 1: Safety checks (always run) ──

        # Battery emergency
        if battery >= 0 and battery < self.MIN_BATTERY_PCT and armed:
            logger.warning(f"LOW BATTERY: {battery}% — forcing RTL")
            self._set_state(FlightState.RETURNING_HOME)
            actions['rtl'] = True
            actions['led_mode'] = self.LED_MODES[FlightState.EMERGENCY]
            actions['report'] = {'event': 'low_battery_rtl', 'battery': battery}
            return actions

        # Altitude ceiling
        if alt > self.MAX_ALTITUDE and armed:
            logger.warning(f"ALTITUDE LIMIT: {alt}m — descending")
            actions['velocity'] = (0, 0, 0.5)  # Descend at 0.5 m/s
            actions['report'] = {'event': 'altitude_limit', 'alt': alt}
            return actions

        # Obstacle avoidance (runs every tick)
        avoidance_result = self.avoidance.update(
            tof_sensors=tof,
            lidar_obstacles=None,  # LiDAR data passed separately if available
            drone_yaw_rad=yaw_rad,
        )

        if avoidance_result['emergency_stop'] and armed:
            logger.warning("EMERGENCY STOP — obstacle too close")
            self._set_state(FlightState.AVOIDING_OBSTACLE)
            actions['hold'] = True
            actions['velocity'] = (0, 0, 0)
            actions['led_mode'] = self.LED_MODES[FlightState.AVOIDING_OBSTACLE]
            actions['report'] = {
                'event': 'emergency_stop',
                'obstacle': avoidance_result['closest_m'],
                'direction': avoidance_result['closest_dir'],
            }
            return actions

        # ── Priority 2: Process cloud commands ──
        if cloud_command:
            self._handle_cloud_command(cloud_command, telemetry, actions)
            if actions['velocity'] or actions['position'] or actions['land'] or actions['rtl']:
                # Cloud command produced an action — apply avoidance and return
                if actions['velocity']:
                    vx, vy, vz = actions['velocity']
                    safe_vx, safe_vy, safe_vz = self.avoidance.adjust_velocity_command(
                        vx, vy, vz, tof, drone_yaw_rad=yaw_rad
                    )
                    actions['velocity'] = (safe_vx, safe_vy, safe_vz)
                return actions

        # ── Priority 3: State machine ──
        if self.state == FlightState.FOLLOWING_SUBJECT:
            self._tick_follow(telemetry, vision_result, avoidance_result, tof, yaw_rad, actions)

        elif self.state == FlightState.ORBITING:
            self._tick_orbit(telemetry, avoidance_result, tof, yaw_rad, dt, actions)

        elif self.state == FlightState.EXECUTING_MISSION:
            self._tick_mission(telemetry, avoidance_result, tof, yaw_rad, actions)

        elif self.state == FlightState.HOVERING:
            actions['hold'] = True
            actions['led_mode'] = self.LED_MODES[FlightState.HOVERING]

        elif self.state == FlightState.AVOIDING_OBSTACLE:
            # If obstacle cleared, return to previous state
            if avoidance_result['closest_m'] > 2.0:
                self._set_state(self.prev_state)
            else:
                # Apply avoidance correction as velocity
                corr = avoidance_result['correction']
                actions['velocity'] = corr
                actions['led_mode'] = self.LED_MODES[FlightState.AVOIDING_OBSTACLE]

        # Set LED based on state
        if actions['led_mode'] is None:
            actions['led_mode'] = self.LED_MODES.get(self.state)

        # Build status report
        actions['report'] = {
            'state': self.state,
            'avoidance': self.avoidance.get_status(),
            'battery': battery,
            'alt': alt,
        }

        return actions

    def _handle_cloud_command(self, cmd, telemetry, actions):
        """Process a command from the cloud/app."""
        action = cmd.get('action', cmd.get('type', '')).upper()
        payload = cmd.get('payload', cmd)

        if action == 'FOLLOW':
            self._set_state(FlightState.FOLLOWING_SUBJECT)
            self.follow_target_id = payload.get('target_id', None)
            self.follow_lost_frames = 0
            logger.info("FOLLOW mode activated")

        elif action == 'ORBIT':
            self._set_state(FlightState.ORBITING)
            self.orbit_center = (
                payload.get('lat', telemetry.get('lat', 0)),
                payload.get('lon', telemetry.get('lon', 0)),
            )
            self.orbit_angle = 0.0
            logger.info(f"ORBIT mode: center={self.orbit_center}")

        elif action == 'GOTO':
            lat = payload.get('lat', 0)
            lon = payload.get('lng', payload.get('lon', 0))
            alt = payload.get('alt', 5)
            actions['position'] = (lat, lon, alt)
            self._set_state(FlightState.EXECUTING_MISSION)

        elif action == 'VELOCITY':
            vx = payload.get('vx', 0)
            vy = payload.get('vy', 0)
            vz = payload.get('vz', 0)
            actions['velocity'] = (vx, vy, vz)

        elif action == 'HOVER' or action == 'HOLD':
            self._set_state(FlightState.HOVERING)
            actions['hold'] = True

        elif action == 'MISSION':
            self.mission_waypoints = payload.get('waypoints', [])
            self.current_waypoint_idx = 0
            if self.mission_waypoints:
                self._set_state(FlightState.EXECUTING_MISSION)
                logger.info(f"MISSION loaded: {len(self.mission_waypoints)} waypoints")

        elif action == 'GIMBAL':
            pitch = payload.get('pitch', 0)
            yaw = payload.get('yaw', 0)
            actions['gimbal'] = (pitch, yaw)

    def _tick_follow(self, telemetry, vision_result, avoidance_result, tof, yaw_rad, actions):
        """Subject following logic."""
        if vision_result is None or not vision_result.get('tracked'):
            self.follow_lost_frames += 1
            if self.follow_lost_frames > 60:  # ~3 seconds at 20Hz
                logger.warning("Follow target lost for 3s — hovering")
                self._set_state(FlightState.HOVERING)
                actions['hold'] = True
            return

        # Find target in tracked objects
        tracked = vision_result['tracked']
        subject = None

        if self.follow_target_id and self.follow_target_id in tracked:
            det = tracked[self.follow_target_id]
            subject = (det.center[0], det.center[1], det.area)
        else:
            # Find largest person
            if self.vision:
                pos = self.vision.get_subject_position(tracked, 'person')
                if pos:
                    subject = (pos[0], pos[1], pos[2])
                    self.follow_target_id = pos[3]

        if subject is None:
            self.follow_lost_frames += 1
            return

        self.follow_lost_frames = 0
        cx, cy, area = subject

        # Frame dimensions (assume 640x480)
        frame_w, frame_h = 640, 480
        center_x, center_y = frame_w / 2, frame_h / 2

        # Error from center (normalized -1 to 1)
        err_x = (cx - center_x) / center_x  # positive = subject is right
        err_y = (cy - center_y) / center_y  # positive = subject is below

        # Yaw to keep subject centered
        yaw_correction = err_x * 20  # degrees

        # Forward/back based on subject size (larger = closer)
        target_area = 15000  # target subject area in pixels
        area_err = (area - target_area) / target_area
        forward_speed = -area_err * self.FOLLOW_SPEED  # negative err = too far = go forward

        # Clamp
        forward_speed = max(-self.FOLLOW_SPEED, min(self.FOLLOW_SPEED, forward_speed))

        # Convert body-frame forward to NED
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        vx = forward_speed * cos_yaw
        vy = forward_speed * sin_yaw

        # Altitude: maintain follow altitude
        alt = telemetry.get('alt_rel', telemetry.get('altitude', 0))
        vz = (self.FOLLOW_ALTITUDE - alt) * -0.3  # negative vz = ascend

        # Apply obstacle avoidance
        safe_vx, safe_vy, safe_vz = self.avoidance.adjust_velocity_command(
            vx, vy, vz, tof, drone_yaw_rad=yaw_rad
        )

        actions['velocity'] = (safe_vx, safe_vy, safe_vz)
        actions['yaw'] = yaw_correction
        actions['gimbal'] = (-err_y * 30, 0)  # Tilt gimbal to track vertically
        actions['led_mode'] = self.LED_MODES[FlightState.FOLLOWING_SUBJECT]

    def _tick_orbit(self, telemetry, avoidance_result, tof, yaw_rad, dt, actions):
        """Orbit logic — fly in a circle around orbit_center."""
        if self.orbit_center is None:
            self._set_state(FlightState.HOVERING)
            return

        # Advance orbit angle
        angular_speed = self.ORBIT_SPEED / self.ORBIT_RADIUS  # rad/s
        self.orbit_angle += angular_speed * dt

        # Target position on circle
        center_lat, center_lon = self.orbit_center
        dlat = self.ORBIT_RADIUS * math.cos(self.orbit_angle) / 111320.0
        dlon = self.ORBIT_RADIUS * math.sin(self.orbit_angle) / (
            111320.0 * math.cos(math.radians(center_lat))
        )
        target_lat = center_lat + dlat
        target_lon = center_lon + dlon

        # Velocity towards target point
        curr_lat = telemetry.get('lat', 0)
        curr_lon = telemetry.get('lon', 0)
        err_lat = (target_lat - curr_lat) * 111320.0  # meters North
        err_lon = (target_lon - curr_lon) * 111320.0 * math.cos(math.radians(curr_lat))  # meters East

        # P controller
        vx = err_lat * 0.8
        vy = err_lon * 0.8

        # Clamp speed
        speed = math.sqrt(vx**2 + vy**2)
        if speed > self.ORBIT_SPEED:
            scale = self.ORBIT_SPEED / speed
            vx *= scale
            vy *= scale

        # Maintain altitude
        alt = telemetry.get('alt_rel', telemetry.get('altitude', 0))
        vz = (self.FOLLOW_ALTITUDE - alt) * -0.3

        # Apply avoidance
        safe_vx, safe_vy, safe_vz = self.avoidance.adjust_velocity_command(
            vx, vy, vz, tof, drone_yaw_rad=yaw_rad
        )

        # Face center
        bearing = math.degrees(math.atan2(
            center_lon - curr_lon,
            center_lat - curr_lat
        )) % 360

        actions['velocity'] = (safe_vx, safe_vy, safe_vz)
        actions['yaw'] = bearing
        actions['gimbal'] = (-15, 0)  # Slight downward tilt
        actions['led_mode'] = self.LED_MODES[FlightState.ORBITING]

    def _tick_mission(self, telemetry, avoidance_result, tof, yaw_rad, actions):
        """Waypoint mission execution."""
        if self.current_waypoint_idx >= len(self.mission_waypoints):
            logger.info("Mission complete — hovering")
            self._set_state(FlightState.HOVERING)
            actions['hold'] = True
            return

        wp = self.mission_waypoints[self.current_waypoint_idx]
        target_lat = wp.get('lat', 0)
        target_lon = wp.get('lng', wp.get('lon', 0))
        target_alt = wp.get('alt', 5)

        curr_lat = telemetry.get('lat', 0)
        curr_lon = telemetry.get('lon', 0)
        curr_alt = telemetry.get('alt_rel', telemetry.get('altitude', 0))

        # Distance to waypoint
        err_lat = (target_lat - curr_lat) * 111320.0
        err_lon = (target_lon - curr_lon) * 111320.0 * math.cos(math.radians(curr_lat))
        dist = math.sqrt(err_lat**2 + err_lon**2)

        # Check if reached
        if dist < 2.0 and abs(curr_alt - target_alt) < 1.5:
            logger.info(f"Waypoint {self.current_waypoint_idx} reached")
            self.current_waypoint_idx += 1
            return

        # Velocity towards waypoint
        if dist > 0.1:
            vx = (err_lat / dist) * min(self.MAX_SPEED, dist * 0.5)
            vy = (err_lon / dist) * min(self.MAX_SPEED, dist * 0.5)
        else:
            vx, vy = 0, 0

        vz = (target_alt - curr_alt) * -0.3

        # Apply avoidance
        safe_vx, safe_vy, safe_vz = self.avoidance.adjust_velocity_command(
            vx, vy, vz, tof, drone_yaw_rad=yaw_rad
        )

        actions['velocity'] = (safe_vx, safe_vy, safe_vz)
        actions['led_mode'] = self.LED_MODES[FlightState.EXECUTING_MISSION]
        actions['report']['waypoint'] = self.current_waypoint_idx
        actions['report']['distance'] = round(dist, 1)

    def get_state(self):
        """Return current state for external queries."""
        return {
            'state': self.state,
            'state_duration': time.time() - self.state_entered_at,
            'mission_progress': (
                f"{self.current_waypoint_idx}/{len(self.mission_waypoints)}"
                if self.mission_waypoints else None
            ),
        }
