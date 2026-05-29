"""
raxda/ai/avoidance.py — Onboard obstacle avoidance engine for Radxa Zero 3W.

Runs locally on the drone at ~20Hz. Uses:
- 4x VL53L1X ToF sensors (front/right/back/left) via ESP32 bridge
- YDLidar X2 360° scan (if available)
- Current velocity + heading from MAVLink telemetry

Produces velocity corrections that steer the drone away from obstacles
while maintaining the intended flight direction as closely as possible.

Data Flow:
    ESP32 (ToF) ──┐
                   ├──▶ ObstacleAvoidance.update() ──▶ velocity_correction (NED)
    LiDAR scan ───┘                                    ▶ should_stop (bool)
                                                       ▶ adjusted path for AI planner
"""
import math
import time
import logging

logger = logging.getLogger('avoidance')


class ObstacleAvoidance:
    """
    Real-time obstacle avoidance running onboard the Radxa Zero 3W.

    Sensor layout (top-down, drone facing North):
        t1 = Front
        t2 = Right
        t3 = Back
        t4 = Left

    Strategy:
    1. STOP zone (<0.5m): Emergency halt, hold position
    2. AVOID zone (0.5-1.5m): Generate repulsive velocity vector
    3. SLOW zone (1.5-3.0m): Scale down commanded velocity
    4. CLEAR zone (>3.0m): No modification

    Uses potential field method: each obstacle creates a repulsive force
    inversely proportional to distance squared. Forces are summed and
    converted to a velocity correction vector.
    """

    # Distance thresholds (meters)
    STOP_DIST = 0.5
    AVOID_DIST = 1.5
    SLOW_DIST = 3.0

    # Maximum avoidance velocity (m/s) — don't overcorrect
    MAX_AVOID_SPEED = 1.5

    # Repulsive force gain — higher = stronger avoidance
    REPULSIVE_GAIN = 0.8

    # ToF sensor directions (angle from North, clockwise, degrees)
    TOF_ANGLES = {
        't1': 0,    # Front → North
        't2': 90,   # Right → East
        't3': 180,  # Back → South
        't4': 270,  # Left → West
    }

    def __init__(self):
        self.last_correction = (0.0, 0.0, 0.0)  # vx, vy, vz correction
        self.emergency_stop = False

        # Predictive tracking — stores previous distances to estimate velocities
        self._prev_distances = {}    # sensor_key -> previous distance
        self._prev_time = 0          # timestamp of previous update
        self._closing_rates = {}     # sensor_key -> m/s (negative = approaching)
        self.closest_obstacle = 9.9
        self.closest_direction = 'none'
        self.last_update = 0.0

        # History for temporal smoothing (prevent oscillation)
        self._correction_history = []
        self._max_history = 5

    def update(self, tof_sensors, lidar_obstacles=None, drone_yaw_rad=0.0):
        """
        Main update loop. Call at 10-20Hz.

        Args:
            tof_sensors: dict {'t1': mm, 't2': mm, 't3': mm, 't4': mm}
                         Values in MILLIMETERS from ESP32. -1 = invalid.
            lidar_obstacles: list of (x, y) in meters, drone-relative frame.
                            None if LiDAR not available.
            drone_yaw_rad: Current heading from MAVLink ATTITUDE (radians).

        Returns:
            dict {
                'correction': (vx, vy, vz),  # NED velocity correction (m/s)
                'emergency_stop': bool,        # True = halt immediately
                'speed_scale': float,          # 0.0-1.0 multiplier for commanded speed
                'closest_m': float,            # Closest obstacle distance
                'closest_dir': str,            # Direction of closest obstacle
            }
        """
        self.last_update = time.time()

        # ── Step 0: Predictive tracking — estimate closing rates ──
        now = time.time()
        dt = now - self._prev_time if self._prev_time > 0 else 0.1
        self._prev_time = now

        for sensor_key in self.TOF_ANGLES:
            dist_mm = tof_sensors.get(sensor_key, -1)
            if dist_mm > 0 and sensor_key in self._prev_distances:
                prev_mm = self._prev_distances[sensor_key]
                if prev_mm > 0 and dt > 0:
                    # Closing rate in m/s (negative = getting closer)
                    closing_rate = (dist_mm - prev_mm) / 1000.0 / dt
                    self._closing_rates[sensor_key] = closing_rate
            self._prev_distances[sensor_key] = dist_mm

        # ── Step 1: Convert ToF readings to obstacles in body frame ──
        obstacles_body = []  # list of (distance_m, angle_rad_body)

        for sensor_key, angle_deg in self.TOF_ANGLES.items():
            dist_mm = tof_sensors.get(sensor_key, -1)
            if dist_mm <= 0 or dist_mm > 8000:  # Invalid or out of range
                continue
            dist_m = dist_mm / 1000.0

            # PREDICTIVE: If obstacle is approaching fast, treat it as closer
            # "That ball/bird/car is coming at 2m/s, in 1 second it'll be here"
            closing = self._closing_rates.get(sensor_key, 0)
            if closing < -0.3:  # Approaching faster than 0.3 m/s
                # Predict where it will be in 1 second
                predicted_dist = dist_m + closing * 1.0  # closing is negative
                dist_m = max(0.1, min(dist_m, predicted_dist))

            angle_rad = math.radians(angle_deg)
            obstacles_body.append((dist_m, angle_rad, sensor_key))

        # ── Step 2: Add LiDAR obstacles (already in body-relative x,y) ──
        if lidar_obstacles:
            for x, y in lidar_obstacles:
                dist_m = math.sqrt(x**2 + y**2)
                if dist_m < 0.1 or dist_m > self.SLOW_DIST:
                    continue
                angle_rad = math.atan2(y, x)
                obstacles_body.append((dist_m, angle_rad, 'lidar'))

        # ── Step 3: Find closest obstacle ──
        if obstacles_body:
            closest = min(obstacles_body, key=lambda o: o[0])
            self.closest_obstacle = closest[0]
            self.closest_direction = closest[2]
        else:
            self.closest_obstacle = 9.9
            self.closest_direction = 'none'

        # ── Step 4: Emergency stop check ──
        if self.closest_obstacle < self.STOP_DIST:
            self.emergency_stop = True
            self.last_correction = (0.0, 0.0, 0.0)
            return {
                'correction': (0.0, 0.0, 0.0),
                'emergency_stop': True,
                'speed_scale': 0.0,
                'closest_m': self.closest_obstacle,
                'closest_dir': self.closest_direction,
            }

        self.emergency_stop = False

        # ── Step 5: Calculate repulsive force field ──
        # Sum repulsive vectors from all obstacles in the AVOID zone
        fx_body = 0.0  # Forward (body frame)
        fy_body = 0.0  # Right (body frame)

        for dist_m, angle_rad, source in obstacles_body:
            if dist_m >= self.AVOID_DIST:
                continue

            # Repulsive magnitude: inversely proportional to distance squared
            # Clamped to prevent division explosion near zero
            safe_dist = max(dist_m, 0.1)
            magnitude = self.REPULSIVE_GAIN * (1.0 / safe_dist - 1.0 / self.AVOID_DIST)

            # Repulsive direction: AWAY from obstacle (opposite of obstacle angle)
            repulse_angle = angle_rad + math.pi  # 180° opposite

            fx_body += magnitude * math.cos(repulse_angle)
            fy_body += magnitude * math.sin(repulse_angle)

        # ── Step 6: Clamp correction magnitude ──
        mag = math.sqrt(fx_body**2 + fy_body**2)
        if mag > self.MAX_AVOID_SPEED:
            scale = self.MAX_AVOID_SPEED / mag
            fx_body *= scale
            fy_body *= scale

        # ── Step 7: Rotate from body frame to NED frame ──
        # Body-forward = drone's heading direction
        cos_yaw = math.cos(drone_yaw_rad)
        sin_yaw = math.sin(drone_yaw_rad)
        vx_ned = fx_body * cos_yaw - fy_body * sin_yaw  # North
        vy_ned = fx_body * sin_yaw + fy_body * cos_yaw  # East

        # No vertical correction from horizontal sensors
        vz_ned = 0.0

        # ── Step 8: Temporal smoothing ──
        self._correction_history.append((vx_ned, vy_ned, vz_ned))
        if len(self._correction_history) > self._max_history:
            self._correction_history.pop(0)

        # Weighted average (more recent = more weight)
        total_w = 0
        avg_vx, avg_vy, avg_vz = 0.0, 0.0, 0.0
        for i, (cvx, cvy, cvz) in enumerate(self._correction_history):
            w = i + 1  # Linear weight: 1, 2, 3, ...
            avg_vx += cvx * w
            avg_vy += cvy * w
            avg_vz += cvz * w
            total_w += w

        if total_w > 0:
            avg_vx /= total_w
            avg_vy /= total_w
            avg_vz /= total_w

        self.last_correction = (avg_vx, avg_vy, avg_vz)

        # ── Step 9: Speed scaling for SLOW zone ──
        if self.closest_obstacle < self.SLOW_DIST:
            # Linear scale: AVOID_DIST → 0.3x, SLOW_DIST → 1.0x
            speed_scale = 0.3 + 0.7 * (
                (self.closest_obstacle - self.AVOID_DIST) /
                (self.SLOW_DIST - self.AVOID_DIST)
            )
            speed_scale = max(0.0, min(1.0, speed_scale))
        else:
            speed_scale = 1.0

        return {
            'correction': (avg_vx, avg_vy, avg_vz),
            'emergency_stop': False,
            'speed_scale': speed_scale,
            'closest_m': self.closest_obstacle,
            'closest_dir': self.closest_direction,
        }

    def adjust_velocity_command(self, commanded_vx, commanded_vy, commanded_vz,
                                 tof_sensors, lidar_obstacles=None, drone_yaw_rad=0.0):
        """
        High-level API: takes a commanded velocity and returns a safe velocity.
        Used by the decision engine to modify AI-planned velocities.

        Args:
            commanded_vx/vy/vz: Desired velocity from AI planner (NED, m/s)
            tof_sensors, lidar_obstacles, drone_yaw_rad: Same as update()

        Returns:
            (safe_vx, safe_vy, safe_vz): Adjusted velocity that avoids obstacles
        """
        result = self.update(tof_sensors, lidar_obstacles, drone_yaw_rad)

        if result['emergency_stop']:
            return (0.0, 0.0, 0.0)

        # Scale commanded velocity
        scale = result['speed_scale']
        scaled_vx = commanded_vx * scale
        scaled_vy = commanded_vy * scale
        scaled_vz = commanded_vz * scale

        # Add avoidance correction
        corr = result['correction']
        safe_vx = scaled_vx + corr[0]
        safe_vy = scaled_vy + corr[1]
        safe_vz = scaled_vz + corr[2]

        return (safe_vx, safe_vy, safe_vz)

    def get_status(self):
        """Returns current avoidance status for telemetry/debugging."""
        return {
            'emergency_stop': self.emergency_stop,
            'closest_m': round(self.closest_obstacle, 2),
            'closest_dir': self.closest_direction,
            'correction': tuple(round(c, 3) for c in self.last_correction),
            'last_update': self.last_update,
            'closing_rates': {k: round(v, 2) for k, v in self._closing_rates.items()},
            'approaching': {k: v for k, v in self._closing_rates.items() if v < -0.3},
        }
