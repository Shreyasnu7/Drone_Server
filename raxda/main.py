"""
raxda/main.py — Main onboard agent for the Radxa Zero 3W.

This is the BRAIN on the drone. It runs the full onboard AI loop:
1. Connect to Flight Controller via MAVLink
2. Connect to cloud server for commands
3. Start heartbeat + auth
4. Initialize obstacle avoidance (ToF + LiDAR)
5. Initialize onboard vision (camera + YOLO)
6. Run the decision engine at ~10Hz

When the laptop AI is connected, the Radxa acts as a bridge.
When disconnected, the Radxa's onboard AI takes over for safety.
"""
import time
import logging
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mavlink.connection import connect
from mavlink.commands import arm, takeoff, land
from mavlink.telemetry import RaxdaTelemetry
from mavlink.failsafe import FailsafeManager
from cloud.client import get_command
from cloud.heartbeat import CloudHeartbeat
from cloud.auth import DroneAuth
from ai.avoidance import ObstacleAvoidance
from ai.decision import DecisionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger('raxda.main')

DRONE_ID = os.environ.get('DRONE_ID', 'RADXA_X')
CLOUD_URL = os.environ.get('CLOUD_URL', 'https://drone-server-r0qe.onrender.com')

# ── Optional: Vision (may not be available on all configs) ──
vision = None
try:
    from ai.vision import VisionProcessor
    vision = VisionProcessor()
    if vision.start_camera():
        logger.info("Onboard vision: ACTIVE")
    else:
        logger.warning("Onboard vision: No camera found")
        vision = None
except Exception as e:
    logger.warning(f"Onboard vision: Disabled ({e})")

# ── Optional: ESP32 sensor bridge ──
esp32 = None
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'raxda_bridge'))
    from esp32_driver import ESP32Driver
    esp32 = ESP32Driver()
    logger.info("ESP32 sensor bridge: ACTIVE")
except Exception as e:
    logger.warning(f"ESP32 bridge: Disabled ({e})")


def main():
    # ── 1. Connect to Flight Controller ──
    logger.info("Connecting to Flight Controller...")
    vehicle = connect()
    logger.info("FC connected!")

    # ── 2. Initialize subsystems ──
    telem = RaxdaTelemetry(vehicle)
    failsafe = FailsafeManager(vehicle)
    avoidance = ObstacleAvoidance()
    decision = DecisionEngine(avoidance, vision)

    # ── 3. Auth + Heartbeat ──
    auth = DroneAuth(DRONE_ID)
    heartbeat = CloudHeartbeat(DRONE_ID, CLOUD_URL)
    heartbeat.start()

    logger.info(f"Radxa Agent ONLINE: {DRONE_ID}")
    logger.info(f"Cloud: {CLOUD_URL}")

    # ── 4. Main loop (10Hz) ──
    loop_hz = 10
    loop_period = 1.0 / loop_hz

    while True:
        loop_start = time.time()

        try:
            # Read FC telemetry (non-blocking)
            telem.update()
            state = telem.get_state()

            # Update heartbeat with latest telemetry
            heartbeat.update_telemetry(state)

            # Read ESP32 sensors
            tof_sensors = {}
            if esp32:
                tof_sensors = esp32.get_telemetry()

            # Read vision
            vision_result = None
            if vision:
                vision_result = vision.process()

            # Check failsafes FIRST (highest priority)
            failsafe_action = failsafe.check(
                state,
                cloud_connected=heartbeat.is_connected
            )
            if failsafe_action:
                logger.warning(f"FAILSAFE: {failsafe_action['action']} — {failsafe_action['reason']}")
                _execute_failsafe(vehicle, failsafe_action)
                # Update heartbeat FC status
                if telem.is_data_fresh():
                    failsafe.fc_heartbeat()
                # Sleep and continue — failsafe takes priority
                time.sleep(loop_period)
                continue

            # Update failsafe FC heartbeat
            if telem.is_data_fresh():
                failsafe.fc_heartbeat()

            # Check for cloud commands
            cloud_cmd = heartbeat.get_pending_command()
            if cloud_cmd is None:
                cloud_cmd = get_command(DRONE_ID)

            # Run decision engine
            actions = decision.tick(
                telemetry=state,
                cloud_command=cloud_cmd,
                tof_sensors=tof_sensors,
                vision_result=vision_result,
            )

            # Execute actions
            _execute_actions(vehicle, actions, esp32)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")

        # Maintain loop rate
        elapsed = time.time() - loop_start
        sleep_time = loop_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Cleanup
    heartbeat.stop()
    if vision:
        vision.stop_camera()
    if esp32:
        esp32.close()
    logger.info("Radxa Agent stopped.")


def _execute_actions(vehicle, actions, esp32=None):
    """Execute the actions produced by the decision engine."""
    from pymavlink import mavutil

    if actions.get('land'):
        land(vehicle)

    elif actions.get('rtl'):
        vehicle.mav.set_mode_send(
            vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6  # RTL
        )

    elif actions.get('hold'):
        vehicle.mav.set_mode_send(
            vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            5  # LOITER
        )

    elif actions.get('velocity'):
        vx, vy, vz = actions['velocity']
        # Send velocity command in NED frame
        vehicle.mav.set_position_target_local_ned_send(
            0, vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000110111000111,  # velocity only
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    elif actions.get('position'):
        lat, lon, alt = actions['position']
        vehicle.mav.set_position_target_global_int_send(
            0, vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    # Yaw
    if actions.get('yaw') is not None:
        vehicle.mav.command_long_send(
            vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            actions['yaw'], 30, 1, 0, 0, 0, 0
        )

    # Gimbal via ESP32
    if actions.get('gimbal') and esp32:
        pitch, yaw = actions['gimbal']
        esp32.set_gimbal(pitch, yaw)

    # LED mode via ESP32
    if actions.get('led_mode') is not None and esp32:
        esp32.set_mode(actions['led_mode'])


def _execute_failsafe(vehicle, failsafe_action):
    """Execute a failsafe action."""
    from pymavlink import mavutil

    action = failsafe_action['action']

    if action == 'LAND':
        land(vehicle)
    elif action == 'RTL':
        vehicle.mav.set_mode_send(
            vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6  # RTL
        )
    elif action == 'DISARM':
        vehicle.mav.command_long_send(
            vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 21196, 0, 0, 0, 0, 0  # Force disarm
        )
    elif action == 'HOLD':
        vehicle.mav.set_mode_send(
            vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            5  # LOITER
        )
    elif action == 'DESCEND':
        vehicle.mav.set_position_target_local_ned_send(
            0, vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000110111000111,
            0, 0, 0,
            0, 0, 0.5,  # Descend at 0.5 m/s
            0, 0, 0,
            0, 0
        )


if __name__ == '__main__':
    main()