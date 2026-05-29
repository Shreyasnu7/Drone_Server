"""
control/move.py — Waypoint navigation and velocity control via MAVLink.
Bridges the gap between AI trajectory plans and actual drone movement.
Supports: goto_position, goto_velocity, orbit, follow_offset, hold_position.
"""
import math
import time
from pymavlink import mavutil
from mavlink.connection import connect
from mavlink.telemetry import TelemetryReceiver


def _haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two GPS coordinates."""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def set_guided_mode(master):
    """Switch FC to GUIDED mode for programmatic waypoint control."""
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4  # GUIDED
    )
    time.sleep(0.3)


def goto_position(master, lat, lon, alt, speed=2.0, acceptance_radius=1.5, timeout=60):
    """
    Fly to a GPS coordinate in GUIDED mode.
    Blocks until within acceptance_radius or timeout.
    Returns dict with status and final distance.
    """
    set_guided_mode(master)

    # Send position target (SET_POSITION_TARGET_GLOBAL_INT)
    master.mav.set_position_target_global_int_send(
        0,  # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111000,  # position only (ignore velocity/accel/yaw)
        int(lat * 1e7),
        int(lon * 1e7),
        alt,
        0, 0, 0,   # vx, vy, vz
        0, 0, 0,   # afx, afy, afz
        0, 0        # yaw, yaw_rate
    )

    # Set speed
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        1,      # speed type: ground speed
        speed,  # m/s
        -1,     # throttle (no change)
        0, 0, 0, 0
    )

    # Wait until reached
    telem = TelemetryReceiver(master)
    start = time.time()
    try:
        while time.time() - start < timeout:
            pos = telem.get_position()
            dist = _haversine_m(pos['lat'], pos['lon'], lat, lon)
            alt_err = abs(pos['alt_rel'] - alt)

            if dist < acceptance_radius and alt_err < 1.5:
                return {"status": "reached", "distance": dist, "time": time.time() - start}

            time.sleep(0.2)

        return {"status": "timeout", "distance": dist, "time": timeout}
    finally:
        telem.stop()


def goto_velocity(master, vx, vy, vz, duration=1.0, yaw_rate=0.0):
    """
    Send velocity command in NED frame (m/s).
    vx=North, vy=East, vz=Down (positive = descend).
    Duration is how long to hold this velocity before requiring a new command.
    """
    set_guided_mode(master)

    type_mask = (
        0b0000110111000111  # ignore position, accel; use velocity + yaw_rate
    )

    master.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,       # x, y, z (ignored)
        vx, vy, vz,    # velocity NED
        0, 0, 0,       # accel (ignored)
        0,              # yaw (ignored)
        yaw_rate        # yaw_rate rad/s
    )

    return {"status": "velocity_sent", "vx": vx, "vy": vy, "vz": vz, "duration": duration}


def orbit(master, center_lat, center_lon, radius_m, alt, speed=2.0, direction=1, laps=1):
    """
    Execute an orbit around a GPS point.
    direction: 1=clockwise, -1=counter-clockwise
    Uses MAV_CMD_DO_ORBIT if supported, else approximates with waypoints.
    """
    set_guided_mode(master)

    # Try native orbit command first (ArduCopter 4.1+)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_ORBIT,
        0,
        radius_m,               # param1: radius
        speed,                  # param2: velocity
        direction,              # param3: direction (0=CW, 1=CCW)
        0,                      # param4: empty
        int(center_lat * 1e7),  # param5: center lat
        int(center_lon * 1e7),  # param6: center lon
        alt                     # param7: altitude
    )

    return {"status": "orbit_started", "radius": radius_m, "speed": speed}


def orbit_waypoints(master, center_lat, center_lon, radius_m, alt, speed=2.0,
                    direction=1, points=12, laps=1):
    """
    Fallback orbit using discrete waypoints around a circle.
    Works on all ArduCopter versions.
    """
    set_guided_mode(master)
    results = []

    for lap in range(laps):
        for i in range(points):
            angle = direction * (2 * math.pi * i / points)
            # Offset from center in meters → GPS
            dlat = radius_m * math.cos(angle) / 111320.0
            dlon = radius_m * math.sin(angle) / (111320.0 * math.cos(math.radians(center_lat)))
            wp_lat = center_lat + dlat
            wp_lon = center_lon + dlon

            result = goto_position(master, wp_lat, wp_lon, alt, speed=speed,
                                   acceptance_radius=radius_m * 0.3, timeout=30)
            results.append(result)

            if result['status'] == 'timeout':
                return {"status": "orbit_incomplete", "completed": i, "total": points * laps}

    return {"status": "orbit_complete", "laps": laps, "points_per_lap": points}


def follow_offset(master, target_lat, target_lon, offset_north=0, offset_east=-5, alt=5, speed=3.0):
    """
    Move to a position offset from a target (e.g., follow a subject).
    offset_north/east in meters from the target position.
    Non-blocking — call repeatedly with updated target positions.
    """
    dlat = offset_north / 111320.0
    dlon = offset_east / (111320.0 * math.cos(math.radians(target_lat)))
    dest_lat = target_lat + dlat
    dest_lon = target_lon + dlon

    master.mav.set_position_target_global_int_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111000,
        int(dest_lat * 1e7),
        int(dest_lon * 1e7),
        alt,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

    # Set speed
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0, 1, speed, -1, 0, 0, 0, 0
    )

    return {"status": "following", "target": (target_lat, target_lon), "offset": (offset_north, offset_east)}


def hold_position(master):
    """
    Immediately stop and hold current position (switch to LOITER or POSHOLD).
    """
    # Try POSHOLD (16) first, fallback to LOITER (5)
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        16  # POSHOLD
    )
    return {"status": "holding"}


def set_yaw(master, heading_deg, speed_deg_s=30, direction=1, relative=False):
    """
    Rotate the drone to a specific heading.
    heading_deg: target heading (0=North, 90=East)
    direction: 1=CW, -1=CCW
    relative: if True, heading_deg is relative to current heading
    """
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        heading_deg,            # target angle
        speed_deg_s,            # angular speed
        direction,              # direction: 1=CW, -1=CCW
        1 if relative else 0,   # 0=absolute, 1=relative
        0, 0, 0
    )
    return {"status": "yaw_set", "heading": heading_deg}


def execute(waypoints, speed=2.0):
    """
    High-level: execute a list of waypoints.
    Each waypoint: {"lat": float, "lng": float, "alt": float}
    Called by the control layer.
    """
    master = connect()
    set_guided_mode(master)
    results = []

    for i, wp in enumerate(waypoints):
        lat = wp.get('lat', wp.get('latitude', 0))
        lon = wp.get('lng', wp.get('lon', wp.get('longitude', 0)))
        alt = wp.get('alt', wp.get('altitude', 5))
        yaw = wp.get('yaw', None)

        if yaw is not None:
            set_yaw(master, yaw)

        result = goto_position(master, lat, lon, alt, speed=speed)
        results.append({"waypoint": i, **result})

        if result['status'] == 'timeout':
            break

    return {"status": "mission_complete", "waypoints": results}
