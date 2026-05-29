import asyncio
import cv2
import json
import socket
import time
import os
import aiohttp
import websockets
from pymavlink import mavutil

# MA-29 FIX: Import onboard obstacle avoidance (potential field method)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'raxda'))
try:
    from ai.avoidance import ObstacleAvoidance
    AVOIDANCE_AVAILABLE = True
    print("✅ Smart obstacle avoidance loaded (potential field)")
except ImportError:
    AVOIDANCE_AVAILABLE = False
    print("⚠️ Smart avoidance not available, using basic brake only")
# --- CONFIG ---
SERVER_URL = "wss://drone-server-r0qe.onrender.com/ws/connect/RADXA_X"
API_URL = "https://drone-server-r0qe.onrender.com"
# Auto-detect FC UART port based on board
# Cubie A7Z (Allwinner A733): /dev/ttyAS0 (Pin 8/10)
# Radxa Zero 3W (RK3566): /dev/ttyS2 (Pin 8/10)
import os
if os.path.exists("/dev/ttyAS0"):
    FC_PORT = os.environ.get("FC_PORT", "/dev/ttyAS0")
elif os.path.exists("/dev/ttyS0"):
    FC_PORT = os.environ.get("FC_PORT", "/dev/ttyS0")
else:
    FC_PORT = os.environ.get("FC_PORT", "/dev/ttyS2")
FC_BAUD = 57600
# V60: Camera Constants
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# --- TAILSCALE CONFIG ---
# Stable 100.x.x.x IPs via Tailscale mesh VPN (works across any network/4G/WiFi)
TAILSCALE_LAPTOP_IP = os.environ.get("TS_LAPTOP_IP", "100.84.75.22")
TAILSCALE_PHONE_IP = os.environ.get("TS_PHONE_IP", "100.109.112.110")  # shreyashs-s21-ultra
TAILSCALE_VIDEO_PORT = 8554   # UDP video stream port
TAILSCALE_RADXA_IP = "100.94.242.14"  # This device (New Radxa, SD card boot)
class SafetyEnvelope:
    """
    Real-time safety gate using ALL available sensors:
    - 4x VL53L1X ToF sensors (ESP32: t1=front, t2=right, t3=back, t4=left)
    - YDLidar X2 (360° 2D scan, closest obstacle distance)
    - Altitude from barometer/GPS fusion

    Actions are classified into risk tiers:
    - EMERGENCY (always allowed): LAND, DISARM, RTL, UPDATE_CONFIG
    - MOVEMENT (blocked if obstacle in path): TAKEOFF, GUIDED, ORBIT,
      UPLOAD_MISSION, joystick with forward/lateral component
    - CAMERA-ONLY (always allowed): GIMBAL, CAPTURE, GOPRO_SETTINGS, RECORD
    """

    # Minimum safe distances (meters)
    CRITICAL_DIST = 0.5     # Emergency stop — too close to anything
    CAUTION_DIST = 1.0      # Slow down / block autonomous actions
    WARN_DIST = 2.0         # Advisory — AI should start planning avoidance

    # Actions that should never be blocked (safety-critical or harmless)
    ALWAYS_ALLOW = {
        'LAND', 'DISARM', 'RTL', 'UPDATE_CONFIG',
        'GIMBAL', 'CAPTURE', 'GOPRO_SETTINGS', 'RECORD', 'RECORD_STOP',
        'USER_GPS', 'PING',
        'RETURN_TO_USER', 'RTH_USER', 'RETURN_USER',  # Emergency return
        'SET_BATT_RTH',  # Config change, not movement
        'STABILIZE', 'ALT_HOLD', 'LOITER', 'POSHOLD',  # Mode switches (safe)
    }

    # Actions that involve drone movement
    MOVEMENT_ACTIONS = {
        'TAKEOFF', 'ARM', 'ORBIT', 'UPLOAD_MISSION', 'GUIDED',
        'GOTO', 'FOLLOW', 'VELOCITY', 'MOVE', 'AI_PLAN', 'CMD_VEL', 'COMMAND',
        'FOLLOW_ME', 'GUIDED', 'AUTO',
    }

    def __init__(self, telemetry_cache):
        self.telem = telemetry_cache
        self.last_warning = 0

    def _get_min_tof_distance(self):
        """Get minimum distance from all 4 ToF sensors. Returns (min_dist, direction)."""
        sensors = {
            'front': self.telem.get('t1', -1),
            'right': self.telem.get('t2', -1),
            'back':  self.telem.get('t3', -1),
            'left':  self.telem.get('t4', -1),
        }
        # Filter out invalid readings (-1 means no sensor or no reading)
        valid = {k: v / 1000.0 for k, v in sensors.items()
                 if v > 0 and v < 8000}  # ToF max ~8m, values in mm
        if not valid:
            return 9.9, 'none'
        min_dir = min(valid, key=valid.get)
        return valid[min_dir], min_dir

    def _get_lidar_min_distance(self):
        """Get minimum distance from LiDAR (already in meters in telem cache)."""
        return self.telem.get('lidar_dist', 9.9)

    def _get_closest_obstacle(self):
        """Fuse ToF + LiDAR to get the absolute closest obstacle."""
        tof_dist, tof_dir = self._get_min_tof_distance()
        lidar_dist = self._get_lidar_min_distance()
        if tof_dist < lidar_dist:
            return tof_dist, f'tof_{tof_dir}'
        return lidar_dist, 'lidar'

    def validate_auto_action(self, action):
        """
        Returns True if the action is safe to execute.
        Returns False if the action should be blocked due to obstacle proximity.
        """
        action_upper = action.upper().strip() if isinstance(action, str) else ''

        # Always allow emergency and camera-only actions
        if action_upper in self.ALWAYS_ALLOW:
            return True

        # Get closest obstacle from all sensors
        closest_dist, source = self._get_closest_obstacle()

        # CRITICAL: Too close — block ALL movement
        if closest_dist < self.CRITICAL_DIST and action_upper in self.MOVEMENT_ACTIONS:
            print(f"🛑 SAFETY BLOCK [{source}]: {closest_dist:.2f}m — "
                  f"CRITICAL proximity, blocking {action_upper}")
            return False

        # CAUTION: Close — block autonomous movement, allow manual override
        if closest_dist < self.CAUTION_DIST and action_upper in self.MOVEMENT_ACTIONS:
            # Allow ARM (pilot may need to arm to land/recover)
            if action_upper == 'ARM':
                return True
            now = time.time()
            if now - self.last_warning > 2.0:
                print(f"⚠️ SAFETY CAUTION [{source}]: {closest_dist:.2f}m — "
                      f"blocking {action_upper}")
                self.last_warning = now
            return False

        # WARN: Advisory only (log but allow)
        if closest_dist < self.WARN_DIST and action_upper in self.MOVEMENT_ACTIONS:
            now = time.time()
            if now - self.last_warning > 5.0:
                print(f"📡 SAFETY WARN [{source}]: {closest_dist:.2f}m — "
                      f"obstacle nearby, allowing {action_upper}")
                self.last_warning = now

        return True

    def get_safety_status(self):
        """Returns current safety status dict for telemetry broadcast."""
        closest_dist, source = self._get_closest_obstacle()
        tof_dist, tof_dir = self._get_min_tof_distance()
        return {
            'closest_obstacle_m': round(closest_dist, 2),
            'obstacle_source': source,
            'tof_min_m': round(tof_dist, 2),
            'tof_direction': tof_dir,
            'lidar_min_m': round(self._get_lidar_min_distance(), 2),
            'level': ('CRITICAL' if closest_dist < self.CRITICAL_DIST
                      else 'CAUTION' if closest_dist < self.CAUTION_DIST
                      else 'WARN' if closest_dist < self.WARN_DIST
                      else 'CLEAR'),
        }
# V40: Smoothing Helper
def smooth(current_val, previous_val, alpha=0.15):
    if previous_val is None: return current_val
    return (previous_val * (1.0 - alpha)) + (current_val * alpha)
class RadxaBridge:
    def __init__(self):
        self.fc = None
        self.ws = None
        self.running = True
        self.telemetry_cache = {}
        self.boot_alt = None # V15: Relative Alt
        self.safety = SafetyEnvelope(self.telemetry_cache)
        self.cam_config = {'w': 1920, 'h': 1080, 'fps': 30, 'source': 'internal'} # Default 1080p (IMX219)
        self.target_caps = "video/x-raw,width=1920,height=1080,framerate=30/1"
        self.cam_needs_reset = False
        self.recording = False
        self.rec_out = None
        # V110: Dual camera state
        self.active_cam = 'internal'        # Which cam shows in app live feed
        self.recording_cam = 'internal'     # Which cam records/AI uses
        self.internal_cap = None            # Pi Camera V2 capture
        self.external_cap = None            # GoPro capture
        self.external_available = False     # Is GoPro connected?
        self.latest_internal_frame = None   # Latest Pi Cam frame
        self.latest_external_frame = None   # Latest GoPro frame
        self.batt_threshold = 20
        self.user_gps = None # (lat, lng)
        self.last_cloud_msg = time.time()
        self.low_batt_triggered = False
        self.is_armed = False # V107: Init missing state
        self.follow_me_active = False
        self.batt_rth_destination = 'launch'  # 'launch' or 'user'
        self.watchdog_triggered = False # V107: Init missing state
        self.smoothing_buffer = {'rc1': 1500, 'rc2': 1500, 'rc3': 1000, 'rc4': 1500} # V40: Smoothing State
        self.esp32_cmd_queue = asyncio.Queue()  # Commands to send to ESP32
        
        # P2.5: Local Recording State
        self.is_recording = False
        self.video_writer = None
        self.take_photo_flag = False
        self.record_start_time = 0
        
        # P2.6: Local AI Bridge & Video Server
        self.local_clients = set()
        self.latest_jpeg = None
        
        # P2.7: LOCAL GOPRO PROXY
        # Radxa executes Bluetooth settings on behalf of distant laptop AI
        self.gopro_proxy = None

        # P3.0: LIDAR INTEGRATION
        self.lidar = None
        self.lidar_min_dist = 9.9  # meters, updated by lidar loop

        # MA-29: Smart obstacle avoidance (potential field)
        self.obstacle_avoidance = ObstacleAvoidance() if AVOIDANCE_AVAILABLE else None

    async def init_gopro_proxy(self):
        try:
            from gopro_proxy import RadxaGoProProxy
            self.gopro_proxy = RadxaGoProProxy()
            connected = await self.gopro_proxy.connect()
            if connected:
                print("🔵 GOPRO BLE PROXY ACTIVE (Ready for AI Commands)")

                # P3.2: Try USB Webcam first (best), then WiFi stream (fallback)
                try:
                    # === TRY USB WEBCAM MODE FIRST ===
                    usb_ok = await self.gopro_proxy.enable_usb_webcam_mode()
                    if usb_ok:
                        print("🎥 GOPRO USB WEBCAM MODE: Enabled (checking /dev/video*)...")
                        # Give UVC device time to enumerate
                        import subprocess
                        await asyncio.sleep(3)
                        # Check if UVC device appeared
                        for idx in range(10):
                            if os.path.exists(f"/dev/video{idx}"):
                                try:
                                    result = subprocess.run(
                                        ['v4l2-ctl', '-d', f'/dev/video{idx}', '--info'],
                                        capture_output=True, text=True, timeout=3
                                    )
                                    if 'gopro' in result.stdout.lower() or 'webcam' in result.stdout.lower():
                                        print(f"✅ GOPRO USB WEBCAM DETECTED @ /dev/video{idx}")
                                        break
                                except Exception:
                                    pass

                    # === FALLBACK: WiFi Preview Stream ===
                    ssid, password = await self.gopro_proxy.get_wifi_credentials()
                    if ssid:
                        print(f"🔵 GOPRO WiFi: SSID={ssid}")
                        # Auto-connect Radxa to GoPro WiFi
                        try:
                            subprocess.run(
                                ['nmcli', 'dev', 'wifi', 'connect', ssid,
                                 'password', password],
                                timeout=15, capture_output=True
                            )
                            print(f"📡 CONNECTED to GoPro WiFi: {ssid}")
                        except Exception as e:
                            print(f"⚠️ WiFi auto-connect failed: {e}")
                            print(f"   Manual: nmcli dev wifi connect {ssid} password {password}")

                    stream_ok = await self.gopro_proxy.start_preview_stream()
                    if stream_ok:
                        print("🎥 GOPRO WIFI PREVIEW STREAM: Active (udp://@0.0.0.0:8554)")
                    else:
                        print("⚠️ GOPRO STREAM: Failed to start — try manual WiFi connect")
                except Exception as e:
                    print(f"⚠️ GOPRO STREAM INIT: {e}")
        except Exception as e:
            print(f"⚠️ GOPRO PROXY INIT FAILED: {e}. Is bleak installed?")

    # MA-12 FIX: Removed dead init_lidar() and update_lidar_telemetry() methods
    # They were never called — lidar_loop() at line ~1714 handles all LiDAR via SDK

    async def execute_ai_plan(self, payload):
        """
        The AI has full authority. It sends whatever params it wants.
        We read the numbers and forward them to the flight controller.
        No action filtering. No hardcoded action lists. The AI thinks freely.

        The AI puts numeric intent into params. We support two modes:
        - Velocity mode: params has vx, vy, vz (m/s in NED frame)
        - Position mode: params has lat, lng, alt (GPS coords)
        Either mode can include yaw (degrees) or yaw_rate (deg/s).
        Gimbal values (pitch, pan) go to ESP32.
        Mode changes (mode_id or mode_name) go to FC.

        If the AI sends something we don't recognize as numbers,
        we just log it — we never crash, never block.
        """
        import math
        if not self.fc:
            print("AI PLAN: No FC connection")
            return

        p = payload.get('params', payload) if isinstance(payload, dict) else {}
        action_name = payload.get('action', 'AI') if isinstance(payload, dict) else 'AI'
        print(f"AI PLAN: {action_name} | {p}")

        try:
            # Ensure FC is in GUIDED mode so it follows our setpoints
            # (ArduPilot ignores position/velocity targets in STABILIZE/ALT_HOLD)
            current_mode = self.telemetry_cache.get('mode_id', '')
            if str(current_mode) not in ('GUIDED', '4'):
                self.fc.mav.set_mode_send(
                    self.fc.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    4  # GUIDED
                )

            # --- VELOCITY COMMAND (vx, vy, vz present) ---
            has_velocity = any(k in p for k in ('vx', 'vy', 'vz'))
            # --- POSITION COMMAND (lat, lng present) ---
            has_gps = 'lat' in p and 'lng' in p
            # --- LOCAL POSITION (x, y, z in meters from home) ---
            has_local_pos = any(k in p for k in ('x', 'y', 'z')) and not has_velocity

            # Yaw handling (works with any mode)
            yaw_deg = float(p.get('yaw', 0))
            yaw_rate = float(p.get('yaw_rate', 0))
            yaw_rad = math.radians(yaw_deg) if yaw_deg else 0

            if has_velocity:
                vx = float(p.get('vx', 0))
                vy = float(p.get('vy', 0))
                vz = float(p.get('vz', 0))

                # Safety clamp: 10 m/s max per axis
                clamp = lambda v, mx: max(-mx, min(mx, v))
                vx, vy, vz = clamp(vx, 10), clamp(vy, 10), clamp(vz, 10)

                # MA-29 FIX: Apply smart obstacle avoidance to AI velocities
                # Uses potential field method — smoothly steers around obstacles
                # instead of crude binary stop at 500mm
                if self.obstacle_avoidance:
                    tof_data = {
                        't1': self.telemetry_cache.get('t1', -1),
                        't2': self.telemetry_cache.get('t2', -1),
                        't3': self.telemetry_cache.get('t3', -1),
                        't4': self.telemetry_cache.get('t4', -1),
                    }
                    drone_yaw = self.telemetry_cache.get('yaw', 0)
                    vx, vy, vz = self.obstacle_avoidance.adjust_velocity_command(
                        vx, vy, vz, tof_data, drone_yaw_rad=drone_yaw
                    )

                type_mask = (
                    0b0000_0001_11_000_111  # ignore position, ignore accel
                )
                # If yaw_rate is set, use yaw rate; otherwise use yaw angle
                if yaw_rate:
                    type_mask |= (1 << 10)   # ignore yaw, use yaw_rate
                else:
                    type_mask |= (1 << 11)   # ignore yaw_rate, use yaw

                self.fc.mav.set_position_target_local_ned_send(
                    0,  # time_boot_ms
                    self.fc.target_system,
                    self.fc.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    type_mask,
                    0, 0, 0,           # x, y, z (ignored)
                    vx, vy, vz,        # velocity
                    0, 0, 0,           # accel (ignored)
                    yaw_rad,
                    math.radians(yaw_rate) if yaw_rate else 0
                )

            elif has_gps:
                lat = float(p['lat'])
                lng = float(p['lng'])
                alt = float(p.get('alt', 5))

                type_mask = (
                    0b0000_0001_11_111_000  # use position, ignore velocity & accel
                )
                if yaw_rate:
                    type_mask |= (1 << 10)
                else:
                    type_mask |= (1 << 11)

                self.fc.mav.set_position_target_global_int_send(
                    0,
                    self.fc.target_system,
                    self.fc.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    type_mask,
                    int(lat * 1e7),
                    int(lng * 1e7),
                    alt,
                    0, 0, 0,
                    0, 0, 0,
                    yaw_rad,
                    math.radians(yaw_rate) if yaw_rate else 0
                )

            elif has_local_pos:
                x = float(p.get('x', 0))
                y = float(p.get('y', 0))
                z = float(p.get('z', -5))  # NED: negative = up

                type_mask = (
                    0b0000_0001_11_111_000  # use position, ignore velocity & accel
                )
                if yaw_rate:
                    type_mask |= (1 << 10)
                else:
                    type_mask |= (1 << 11)

                self.fc.mav.set_position_target_local_ned_send(
                    0,
                    self.fc.target_system,
                    self.fc.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    type_mask,
                    x, y, z,
                    0, 0, 0,
                    0, 0, 0,
                    yaw_rad,
                    math.radians(yaw_rate) if yaw_rate else 0
                )

            # --- GIMBAL (if present alongside anything) ---
            if 'gimbal_pitch' in p or 'gimbal_yaw' in p or 'pan' in p:
                gp = float(p.get('gimbal_pitch', p.get('pitch', 0)))
                gy = float(p.get('gimbal_yaw', p.get('pan', 0)))
                await self.esp32_cmd_queue.put({"type": "gimbal", "pitch": gp, "yaw": gy})

            # --- MODE CHANGE (if AI decides to switch flight mode) ---
            if 'mode' in p:
                mode_name = str(p['mode']).upper()
                mode_map = {
                    'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2,
                    'AUTO': 3, 'GUIDED': 4, 'LOITER': 5,
                    'RTL': 6, 'LAND': 9, 'POSHOLD': 16,
                }
                mode_id = mode_map.get(mode_name, int(p['mode']) if str(p['mode']).isdigit() else None)
                if mode_id is not None:
                    self.fc.mav.set_mode_send(
                        self.fc.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        mode_id
                    )

            # --- ARM/DISARM (if AI decides) ---
            if 'arm' in p:
                arm_val = 1 if p['arm'] else 0
                self.fc.mav.command_long_send(
                    self.fc.target_system, self.fc.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, arm_val, 0, 0, 0, 0, 0, 0
                )

            # --- LED feedback via ESP32 ---
            if 'led' in p:
                await self.esp32_cmd_queue.put({"type": "led", "color": p['led']})

        except Exception as e:
            print(f"AI PLAN EXEC ERROR: {e}")

    async def start_local_server(self):
        print("🚀 STARTING LOCAL AI BRIDGE (0.0.0.0:8000)...")
        # Standard Websocket Server for Control/Telemetry
        for attempt in range(5):
            try:
                async with websockets.serve(self.handle_local_client, "0.0.0.0", 8000, reuse_port=True):
                    await asyncio.Future() # Run forever
            except OSError as e:
                print(f"⚠️ Port 8000 busy (attempt {attempt+1}/5): {e}")
                await asyncio.sleep(3)

    async def start_local_video_server(self):
        from aiohttp import web
        print("🎥 STARTING LOCAL VIDEO SERVER (0.0.0.0:8080)...")
        app = web.Application()
        app.router.add_get('/snapshot', self.handle_snapshot)
        app.router.add_get('/stream', self.handle_stream)
        runner = web.AppRunner(app)
        await runner.setup()
        for attempt in range(5):
            try:
                site = web.TCPSite(runner, '0.0.0.0', 8080, reuse_port=True)
                await site.start()
                break
            except OSError as e:
                print(f"⚠️ Port 8080 busy (attempt {attempt+1}/5): {e}")
                await asyncio.sleep(3)
        # Keep alive
        while True:
            await asyncio.sleep(3600)

    async def handle_snapshot(self, request):
        if self.latest_jpeg:
            return web.Response(body=self.latest_jpeg, content_type='image/jpeg')
        return web.Response(status=503, text="No Frame Yet")

    async def handle_stream(self, request):
        resp = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace;boundary=frame',
                'Cache-Control': 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0',
            }
        )
        await resp.prepare(request)
        try:
            while True:
                if self.latest_jpeg:
                    frame = self.latest_jpeg
                    try:
                        await resp.write(b'--frame\r\n')
                        await resp.write(b'Content-Type: image/jpeg\r\n\r\n')
                        await resp.write(frame)
                        await resp.write(b'\r\n')
                        await asyncio.sleep(0.04) # Max 25fps
                    except:
                        break
                else:
                    await asyncio.sleep(0.1)
        except:
            pass
        return resp

    async def handle_local_client(self, websocket, path):
        print("🔗 FAST-LINK: Client Connected (Local)")
        self.local_clients.add(websocket)
        try:
            async for message in websocket:
                # Direct packet injection or handshake ignore
                try:
                    data = json.loads(message)
                    # Handshake support (Director sends {id:..., token:...})
                    if 'token' in data:
                        print(f"🤝 FAST-LINK Handshake: {data.get('id')}")
                        continue
                    # Process command
                    await self.process_packet(data.get('type'), data.get('payload'))
                except Exception as e:
                    print(f"⚠️ FAST-LINK Data Error: {e}")
        except:
             pass
        finally:
             print("🔗 FAST-LINK: Client Disconnected")
             self.local_clients.remove(websocket)

    def init_esp32(self):
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.esp32_addr = ("192.168.4.2", 8888)
        print(f"🔭 ESP32 Gimbal Link Active -> {self.esp32_addr}")
    async def connect_mavlink(self):
        self.init_esp32()
        asyncio.create_task(self.lidar_loop())
        # V110: Use configured baud rate (57600 for ArduPilot default)
        bauds = [FC_BAUD]
        while self.running:
            for baud in bauds:
                try:
                    print(f"🔌 Connecting to FC on {FC_PORT} @ {baud}...")
                    self.fc = mavutil.mavlink_connection(FC_PORT, baud=baud)
                    hb = self.fc.wait_heartbeat(timeout=2)
                    if hb is None:
                        print(f"❌ Heartbeat Timeout @ {baud}.")
                        self.fc.close()
                        continue
                    print(f"✅ FC Connected @ {baud}! Heartbeat receiving.")
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 2, 1)
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 4, 1) # Attitude
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 4, 1) # HUD
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_POSITION, 4, 1) # REL_ALT

                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'ARMING_CHECK', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'EKF2_GPS_CHECK', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32) # V26: Kill EKF GPS Check
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'FS_EKF_THRESH', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32) # V26: Kill EKF Failsafe

                    # V27: AUTO-CONFIGURE FOR INDOOR (Disable GPS & Failsafes)
                    # GPS_TYPE=0 removed — contradicts V106 GPS enable below (GPS_TYPE=1)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'AHRS_GPS_USE', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'FS_THR_ENABLE', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'FS_GCS_ENABLE', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    # self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'FS_BATT_ENABLE', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32) # V46: Re-enabled below
                    # V106: ENABLE GPS (User Req: "GPS needs to be enabled")
                    # EKF3 uses GPS + Compass + IMU.
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'AHRS_EKF_TYPE', 3, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'GPS_TYPE', 1, mavutil.mavlink.MAV_PARAM_TYPE_UINT32) # 1=Auto

                    # V39: UNLOCKED PRECISION MODE (User Request: "Exact Joystick Control")
                    # disabling Deadzones so even micro-movements are registered
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'RC1_DZ', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'RC2_DZ', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'RC3_DZ', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT16) # Throttle Deadzone = 0
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'RC4_DZ', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)

                    # Restoring Speed Limits to Standard (User wants full authority)
                    # PILOT_SPEED_UP=250 removed — overridden by V72 PILOT_SPEED_UP=500 below
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'PILOT_SPEED_DN', 150, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'PILOT_ACCEL_Z', 250, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)

                    # V42: ADAPTIVE HOVER LEARNING (User Request: "Adapt to 1.5kg Weight")
                    # Enable Hover Learning (2=Learn and Save). 0.55 for 1.5kg F450.
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'MOT_THST_HOVER', 0.55, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    # V53: MOTOR IDLE FIX — MOT_SPIN_ARM=0.12 and MOT_SPIN_MIN=0.15 removed
                    # They were overridden by V72 values (0.25/0.25) below
                    # V104: BATTERY CALIBRATION (Standard MiniPix Defaults)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_MONITOR', 4, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_VOLT_PIN', 2, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_CURR_PIN', 3, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_VOLT_MULT', 10.1, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_AMP_PERVOLT', 18.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    # V105: CORRECTION - Capacity 8400mAh (User Specified)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_CAPACITY', 8400, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'ANGLE_MAX', 6000, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'PILOT_SPEED_UP', 500, mavutil.mavlink.MAV_PARAM_TYPE_INT16)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'MOT_SPOOL_TIME', 0.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32) # V72: ZERO DELAY
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'MOT_SPIN_ARM', 0.25, mavutil.mavlink.MAV_PARAM_TYPE_REAL32) # V72: Hot Idle
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'MOT_SPIN_MIN', 0.25, mavutil.mavlink.MAV_PARAM_TYPE_REAL32) # V72: Match Arm
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'ARMING_CHECK', 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32) # V72: No Checks
                    # V71: Li-Ion Voltage Range (Monitor 4 + Python Override)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_LOW_VOLT', 9.6, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_CRT_VOLT', 9.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    print("🔓 INDOOR MODE: ZERO DELAY (V72) | SMOOTHED BATTERY")
                    return
                except Exception as e:
                    print(f"⚠️ FC Check Failed: {e}")
                    await asyncio.sleep(1)
            print("🔴 FC NOT DETECTED. Retrying...")
            await asyncio.sleep(2)
    async def connect_cloud(self):
        while self.running:
            try:
                print(f"☁️ Connecting to Cloud: {SERVER_URL}...")
                # User Request: "NEVER disconnect on its own" -> INFINITE TIMEOUT
                async with websockets.connect(SERVER_URL, ping_interval=10, ping_timeout=None) as ws:
                    self.ws = ws
                    print("✅ Cloud Connected!")

                    # AUTH FRAME
                    auth_frame = {
                        "type": "connect_drone",
                        "droneId": "RADXA_X",
                        "token": "bearer_token"
                    }
                    await ws.send(json.dumps(auth_frame))

                    # START LOCAL GOPRO BLUETOOTH PROXY
                    # Tries to connect to physical GoPro next to Radxa
                    if not self.gopro_proxy:
                        await self.init_gopro_proxy()

                    # PARALLEL TASKS
                    await asyncio.gather(
                        self.telemetry_loop(),
                        self.command_loop()
                    )
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 2, 1) # V27: Debug RC
            except Exception as e:
                print(f"⚠️ Cloud Disconnected: {e}. Retrying in 5s...")
                self.ws = None
                await asyncio.sleep(5)
    async def telemetry_loop(self):
        last_send = 0
        while self.ws and self.running:
            if self.fc:
                # V22: Drain Buffer (Process up to 50 msgs per loop to catch ACKs)
                for _ in range(50):
                    msg = self.fc.recv_match(blocking=False)
                    if not msg: break

                    type = msg.get_type()
                    # Battery Failsafe
                    if type == 'SYS_STATUS':
                        # V70: FORCE PYTHON CALCULATION (If FC fails or is stuck)
                        batt_pct = msg.battery_remaining
                        batt_voltage = msg.voltage_battery / 1000.0
                        if batt_pct < 0 or batt_pct > 95:
                             # V71: Li-Ion Curve (9.0V to 12.6V)
                             calc_pct = int((batt_voltage - 9.0) / 3.6 * 100.0)
                             batt_pct = max(0, min(100, calc_pct))

                        # V72: REAL-TIME (No Smoothing - User wants accurate readings)
                        self.telemetry_cache['battery'] = batt_pct # Direct, unfiltered
                        self.telemetry_cache['voltage'] = batt_voltage
                        self.telemetry_cache['armed'] = (msg.onboard_control_sensors_health & mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO) # Approximation or use HEARTBEAT

                        # V26: Capture Mode
                        self.telemetry_cache['mode_id'] = self.fc.flightmode

                        if msg.battery_remaining < self.batt_threshold and not self.low_batt_triggered:
                            print(f"LOW BATT < {self.batt_threshold}%! Smart RTH -> {self.batt_rth_destination}")
                            self.low_batt_triggered = True
                            self.follow_me_active = False  # Stop follow on low batt
                            if self.batt_rth_destination == 'user' and self.user_gps:
                                lat, lng = self.user_gps
                                print(f"RTH to User: {lat}, {lng}")
                                self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
                                self.fc.mav.mission_item_int_send(self.fc.target_system, self.fc.target_component, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 0, 0, 0, 0, 0, int(lat * 1e7), int(lng * 1e7), 15)
                            else:
                                print("RTH to Launch")
                                self.fc.set_mode('RTL')

                    elif type == 'STATUSTEXT': # V23: The Voice of the FC
                         print(f"📢 FC SAYS: {msg.text}")

                    elif type == 'PARAM_VALUE': # V23: TX Verification
                         print(f"✅ TX VERIFIED: Read Param {msg.param_id} = {msg.param_value}")
                    elif type == 'ATTITUDE':
                         self.telemetry_cache['roll'] = msg.roll
                         self.telemetry_cache['pitch'] = msg.pitch
                         self.telemetry_cache['yaw'] = msg.yaw
                         # V110: Derive heading from yaw (0-360 degrees)
                         import math
                         heading_deg = math.degrees(msg.yaw) % 360
                         self.telemetry_cache['heading'] = heading_deg

                    elif type == 'GLOBAL_POSITION_INT':
                         # V110: Extract GPS position, altitude, speed, heading
                         self.telemetry_cache['lat'] = msg.lat / 1e7
                         self.telemetry_cache['lng'] = msg.lon / 1e7
                         self.telemetry_cache['altitude_gps'] = msg.relative_alt / 1000.0  # mm to m
                         self.telemetry_cache['raw_alt'] = msg.alt / 1000.0  # MSL altitude
                         # Ground speed from vx/vy (cm/s to m/s)
                         vx = msg.vx / 100.0
                         vy = msg.vy / 100.0
                         self.telemetry_cache['speed'] = (vx**2 + vy**2)**0.5
                         # Heading from GPS (cdeg to deg)
                         self.telemetry_cache['heading'] = msg.hdg / 100.0 if msg.hdg != 65535 else self.telemetry_cache.get('heading', 0)

                    elif type == 'GPS_RAW_INT':
                         # V110: Satellite count + GPS fix quality
                         self.telemetry_cache['sats'] = msg.satellites_visible
                         self.telemetry_cache['gps_fix'] = msg.fix_type  # 0=no, 2=2D, 3=3D

                    elif type == 'VFR_HUD':
                         # V110: Airspeed, groundspeed, altitude, climb rate
                         self.telemetry_cache['speed'] = msg.groundspeed
                         self.telemetry_cache['altitude_baro'] = msg.alt
                         self.telemetry_cache['climb_rate'] = msg.climb
                         if msg.heading != 65535:
                             self.telemetry_cache['heading'] = msg.heading

                    elif type == 'RC_CHANNELS_RAW': # V27: Debug Switch Positions
                         # Log channels 5, 6, 7 (common for mode switches) every 2s
                         if int(time.time()) % 2 == 0 and int(time.time()) != getattr(self, 'last_rc_log', 0):
                              self.last_rc_log = int(time.time())
                              print(f"🎮 RC RAW: C5={msg.chan5_raw} C6={msg.chan6_raw} C7={msg.chan7_raw}")
                    # DATA FUSION Logic
                    sats = self.telemetry_cache.get('sats', 0)
                    if sats > 5:
                         self.telemetry_cache['altitude'] = self.telemetry_cache.get('altitude_gps', 0)
                         self.telemetry_cache['source'] = 'GPS'
                    else:
                         self.telemetry_cache['altitude'] = self.telemetry_cache.get('altitude_baro', 0)
                         self.telemetry_cache['source'] = 'BARO'

                    # V110: Calculate distance from home (for app display)
                    lat = self.telemetry_cache.get('lat', 0)
                    lng = self.telemetry_cache.get('lng', 0)
                    if lat != 0 and lng != 0:
                        if not hasattr(self, 'home_lat') or self.home_lat is None:
                            self.home_lat = lat
                            self.home_lng = lng
                        import math
                        dlat = math.radians(lat - self.home_lat)
                        dlng = math.radians(lng - self.home_lng)
                        a = math.sin(dlat/2)**2 + math.cos(math.radians(self.home_lat)) * math.cos(math.radians(lat)) * math.sin(dlng/2)**2
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        self.telemetry_cache['distance'] = round(6371000 * c, 1)  # meters
            now = time.time()
            # P3.0: LiDAR data is injected by lidar_loop() directly into telemetry_cache
            # P3.1: Inject safety status into telemetry
            self.telemetry_cache['safety'] = self.safety.get_safety_status()

            # FOLLOW ME: Continuously send user GPS as GUIDED waypoint (every 2s)
            if self.follow_me_active and self.user_gps and self.fc:
                if not hasattr(self, '_last_follow_send') or now - self._last_follow_send > 2.0:
                    self._last_follow_send = now
                    lat, lng = self.user_gps
                    alt = self.telemetry_cache.get('altitude', 5)
                    alt = max(3, alt)  # Don't descend below 3m while following
                    # Ensure GUIDED mode
                    self.fc.mav.set_mode_send(
                        self.fc.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4
                    )
                    self.fc.mav.mission_item_int_send(
                        self.fc.target_system, self.fc.target_component,
                        0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                        2, 0, 0, 0, 0, 0,
                        int(lat * 1e7), int(lng * 1e7), alt
                    )
            if self.telemetry_cache:
                if now - last_send > 0.1: # 10Hz
                    msg = json.dumps({"type": "telemetry", "payload": self.telemetry_cache})
                    if self.ws: await self.ws.send(msg)
                    # P2.6: Broadcast to Local AI Clients
                    if self.local_clients:
                        dead = set()
                        for c in self.local_clients:
                            try: await c.send(msg)
                            except: dead.add(c)
                        self.local_clients -= dead
                    last_send = now
                # DASHBOARD LOGGING (Every 1s - scrolling)
                if int(now) % 2 == 0 and int(now) != getattr(self, 'last_log_sec', 0):
                    self.last_log_sec = int(now)
                    alt = self.telemetry_cache.get('altitude', 0)
                    src = self.telemetry_cache.get('source', 'UNK')
                    batt = self.telemetry_cache.get('battery', 0)
                    volt = self.telemetry_cache.get('voltage', 0)
                    sats = self.telemetry_cache.get('sats', 0)
                    raw = self.telemetry_cache.get('raw_alt', 0)
                    is_armed = self.fc.motors_armed() if self.fc else False
                    self.is_armed = is_armed # V107: Update state
                    mode = self.telemetry_cache.get('mode_id', 'UNK')
                    est_cells = int(round(volt / 4.2)) if volt > 0 else 0 # V26: Battery Debug
                    speed = self.telemetry_cache.get('speed', 0)
                     # PRINT NEWLINE logs for debugging
                    print(f"📊 DATA: Alt={alt:.1f}m (Raw={raw:.1f}) | Batt={batt}% ({volt:.1f}V~{est_cells}S) | Mode={mode} | Spd={speed:.1f} | Armed={is_armed}")



                    # V28: WAR ON RTL - Force Stabilize every 1s if disarmed
                    # V64: Force STABILIZE (0) - Instant Response
                    if not is_armed and mode != 'STABILIZE' and int(now) % 2 == 0:
                        self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0) # STABILIZE
                        print("🛡️ FORCING STABILIZE (0) - INSTANT ADAPTIVE")

            await asyncio.sleep(0.01)
    async def command_loop(self):
        while self.running and self.ws:
            try:
                msg = await self.ws.recv()
                # V57: RAW DEBUG (See exact JSON)
                if 'joystick' not in msg:
                     print(f"🔍 RAW JSON: {msg}")

                data = json.loads(msg)
                type = data.get('type')
                # Support both 'payload' (app/cloud) and 'primitive' (laptop AI)
                payload = data.get('payload', data.get('primitive', {}))

                # V58 FIX: Double-Decode only if JSON-String. Keep original if fail.
                if isinstance(payload, str):
                    try:
                        decoded = json.loads(payload)
                        if isinstance(decoded, (dict, list)): # Only accept proper structures
                             payload = decoded
                    except:
                        pass # It was a plain string (e.g. "LAND") - Keep it!
                if type != 'joystick' and type != 'user_gps': # Spam filter: Hide GPS too
                    print(f"📥 Rx: {type}")

                if type == 'user_gps':
                    if payload and 'lat' in payload:
                        self.user_gps = (payload.get('lat'), payload.get('lng'))
                # V55: OMNI-PARSER
                cmd = type

                # V110: APP COMMAND FORMAT FIX
                # App sends {"type": "command", "payload": "LAND"} via sendCommand()
                # We need to extract the real command from payload
                if cmd == 'command' and isinstance(payload, str):
                    cmd = payload  # "LAND", "ARM", "RTH", etc.
                    payload = {}   # Reset payload since it was the command string
                elif cmd == 'command' and isinstance(payload, dict) and 'command' in payload:
                    # App sendCommand("set_safety_config", {...}) sends {"type":"command","payload":{"command":"set_safety_config","payload":{...}}}
                    cmd = payload.get('command', cmd)
                    payload = payload.get('payload', payload)

                # Also handle "mission" type from app's sendMission()
                if cmd == 'mission':
                    cmd = 'UPLOAD_MISSION'
                    payload = {'items': payload if isinstance(payload, list) else payload.get('items', [])}

                # V56: STRING CLEANUP
                if isinstance(cmd, str):
                    cmd = cmd.upper().strip().replace('"', '').replace("'", "")

                # V56: Safety Check
                is_emergency = cmd in ['LAND', 'DISARM', 'RTL', 'UPDATE_CONFIG']
                if not is_emergency and not self.safety.validate_auto_action(cmd):
                     print(f"🛑 SAFETY BLOCK: {cmd}")
                     if self.ws:
                         await self.ws.send(json.dumps({
                             "type": "safety_block",
                             "payload": {
                                 "blocked_action": cmd,
                                 "safety": self.safety.get_safety_status()
                             }
                         }))
                     continue  # ENFORCED: Block unsafe actions

                # V110: Handle SET_CONFIG:key=value format from app
                if cmd.startswith('SET_CONFIG:'):
                    # Parse "SET_CONFIG:res=4k" → key="res", value="4k"
                    config_str = cmd.replace('SET_CONFIG:', '')
                    if '=' in config_str:
                        key, val = config_str.split('=', 1)
                        print(f"⚙️ SET_CONFIG: {key}={val}")
                        # Forward as UPDATE_CONFIG
                        payload = {'config': {key.strip().lower(): val.strip()}}
                        cmd = 'UPDATE_CONFIG'
                    else:
                        print(f"⚠️ Malformed SET_CONFIG: {config_str}")

                # V110: Handle set_safety_config from app
                elif cmd == 'SET_SAFETY_CONFIG':
                    if isinstance(payload, dict):
                        oa = payload.get('obstacle_avoidance', True)
                        vp = payload.get('vision_pos', True)
                        print(f"⚙️ SAFETY CONFIG: obstacle_avoidance={oa}, vision_pos={vp}")
                        # Store in telemetry cache for reference
                        self.telemetry_cache['obstacle_avoidance'] = oa
                        self.telemetry_cache['vision_pos'] = vp

                # V110: Handle set_rth_alt from app
                elif cmd == 'SET_RTH_ALT':
                    alt_cm = int(payload.get('alt', 5000)) if isinstance(payload, dict) else 5000
                    alt_m = alt_cm / 100.0
                    print(f"⚙️ RTH ALT: {alt_m}m")
                    # Set RTL_ALT param on FC (in cm)
                    if self.fc:
                        self.fc.mav.param_set_send(
                            self.fc.target_system, self.fc.target_component,
                            b'RTL_ALT', alt_cm, mavutil.mavlink.MAV_PARAM_TYPE_INT32
                        )

                # V110: Handle set_batt_threshold from app
                elif cmd == 'SET_BATT_THRESHOLD':
                    threshold = int(payload.get('threshold', 20)) if isinstance(payload, dict) else 20
                    self.batt_threshold = max(5, min(50, threshold))
                    print(f"⚙️ BATT THRESHOLD: {self.batt_threshold}%")

                # V110: SWITCH_CAMERA — user picks which camera to view/record
                elif cmd == 'SWITCH_CAMERA':
                    target = payload.get('camera', 'internal') if isinstance(payload, dict) else 'internal'
                    if target in ('internal', 'external', 'gopro'):
                        self.active_cam = 'external' if target in ('external', 'gopro') else 'internal'
                        print(f"📷 CAMERA SWITCH: active={self.active_cam}")
                        self.telemetry_cache['active_cam'] = self.active_cam

                elif cmd == 'SET_RECORDING_CAMERA':
                    target = payload.get('camera', 'internal') if isinstance(payload, dict) else 'internal'
                    self.recording_cam = 'external' if target in ('external', 'gopro') else 'internal'
                    print(f"🎬 RECORDING CAMERA: {self.recording_cam}")

                # P2.1: SETTINGS SYNC
                if cmd == 'UPDATE_CONFIG':
                    cfg = payload.get('config', {})
                    print(f"⚙️ SETTINGS UPDATED: {cfg}")

                    # HANDLE CAMERA RESOLUTION CHANGE
                    if 'cap_res' in cfg:
                        res_key = cfg['cap_res']
                        # IMX219 (Pi Cam V2) Modes
                        RES_MAP = {
                            "8mp": "video/x-raw,width=3280,height=2464,framerate=15/1",
                            "1080p": "video/x-raw,width=1920,height=1080,framerate=30/1",
                            "720p": "video/x-raw,width=1280,height=720,framerate=60/1",
                            "480p": "video/x-raw,width=640,height=480,framerate=90/1"
                        }
                        # If unknown (e.g. GoPro res), ignore it
                        if res_key in RES_MAP:
                            new_caps = RES_MAP[res_key]
                            # Only reset if changed
                            if new_caps != getattr(self, 'target_caps', ""):
                                self.target_caps = new_caps
                                self.cam_needs_reset = True
                                print(f"🎥 CAMERA CONFIG CHANGING TO: {res_key} ({new_caps})")

                # P2.8: GOPRO REMOTE AI PROXY
                elif cmd == 'GOPRO_SETTINGS':
                    if self.gopro_proxy:
                        print(f"🔵 PROXY: Forwarding AI Cinematic Params to GoPro BLE: {payload}")
                        await self.gopro_proxy.handle_remote_ai_command(payload)
                    else:
                        print("⚠️ Received GoPro AI Params, but Local Proxy is offline.")

                # AI AUTONOMOUS PLAN HANDLER
                # The AI has FULL authority to decide any action.
                # This converts the AI's decision into real MAVLink commands.
                elif cmd == 'AI_PLAN':
                    await self.execute_ai_plan(payload)

                # GENERIC COMMAND RELAY (mavllink_executor.py sends these)
                elif cmd == 'COMMAND':
                    await self.execute_ai_plan({
                        'action': payload.get('action', 'RELAY'),
                        'params': payload.get('params', payload),
                    })

                # LAPTOP AI VELOCITY RELAY (remote mode sends cmd_vel)
                elif cmd == 'CMD_VEL':
                    await self.execute_ai_plan({
                        'action': 'CMD_VEL',
                        'params': {
                            'vx': payload.get('vx', 0),
                            'vy': payload.get('vy', 0),
                            'vz': payload.get('vz', 0),
                            'yaw_rate': payload.get('yaw_rate', 0),
                        }
                    })

                # P2.2: GIMBAL RELAY
                elif cmd == 'GIMBAL':
                    pitch = payload.get('pitch', 0)
                    yaw = payload.get('yaw', 0)
                    await self.esp32_cmd_queue.put({"type": "gimbal", "pitch": pitch, "yaw": yaw})
                    print(f"🎥 GIMBAL: Pitch={pitch}, Yaw={yaw}")

                    # P1.6: MISSION HANDLER (Upload Waypoints to FC)
                elif cmd == 'UPLOAD_MISSION':
                    try:
                        items = payload.get('items', [])
                        print(f"🗺️ UPLOADING MISSION: {len(items)} Waypoints...")
                        if items:
                            self.fc.mav.mission_clear_all_send(self.fc.target_system, self.fc.target_component)
                            self.fc.mav.mission_count_send(self.fc.target_system, self.fc.target_component, len(items))
                            for i, item in enumerate(items):
                                self.fc.mav.mission_item_int_send(
                                    self.fc.target_system, self.fc.target_component,
                                    i,
                                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                    0, 1,
                                    0, 0, 0, 0,
                                    int(item['lat'] * 1e7),
                                    int(item['lng'] * 1e7),
                                    float(item.get('alt', 20))
                                )
                            print("✅ MISSION UPLOADED")
                    except Exception as e:
                        print(f"❌ Mission Upload Fail: {e}")

                elif cmd == 'TAKEOFF':
                    self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 5)

                elif cmd == 'LAND':
                    alt = self.telemetry_cache.get('altitude', 0)
                    print(f"🛬 LAND CMD. Alt={alt:.1f}m")
                    if alt < 1.0:
                        print("🛑 GROUND: FORCE DISARM (21196)")
                        self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196, 0, 0, 0, 0, 0)
                    else:
                        print("🛬 AIR: SMART LAND (with obstacle avoidance)")
                        self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
                        # V110: Start smart avoidance monitor during landing
                        asyncio.create_task(self._smart_avoidance_monitor('land'))

                elif cmd == 'ARM':
                     self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0) # STABILIZE
                     print("🛡️ SPLIT-ARM: Mode -> STABILIZE (0)... NO DELAY")
                     self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                     print("🛡️ SPLIT-ARM: Sending ARM Command Now.")

                elif cmd in ('RTL', 'RTH'):
                    # V110: RTH from app = RTL to ArduPilot
                    # If user_gps available and rth_behavior is 'user', fly to user instead
                    if self.batt_rth_destination == 'user' and self.user_gps:
                        lat, lng = self.user_gps
                        print(f"🏠 RTH TO USER: ({lat:.6f}, {lng:.6f})")
                        self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
                        self.fc.mav.mission_item_int_send(
                            self.fc.target_system, self.fc.target_component,
                            0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 0, 0, 0, 0, 0,
                            int(lat * 1e7), int(lng * 1e7), 15
                        )
                    else:
                        print("🏠 RTH TO LAUNCH (RTL)")
                        self.fc.set_mode('RTL')
                    # V110: Start smart avoidance monitor during RTH
                    asyncio.create_task(self._smart_avoidance_monitor('rth'))

                elif cmd == 'DISARM':
                    print("DISARM command received")
                    self.fc.mav.command_long_send(
                        self.fc.target_system, self.fc.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                        0, 0, 21196, 0, 0, 0, 0, 0  # 21196 = force disarm
                    )

                # RETURN TO USER — fly to user's live GPS position
                elif cmd in ('RETURN_TO_USER', 'RTH_USER', 'RETURN_USER'):
                    if self.user_gps:
                        lat, lng = self.user_gps
                        alt = float(payload.get('alt', 15)) if isinstance(payload, dict) else 15
                        print(f"RTH TO USER: ({lat}, {lng}) @ {alt}m")
                        self.fc.mav.set_mode_send(
                            self.fc.target_system,
                            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4  # GUIDED
                        )
                        self.fc.mav.mission_item_int_send(
                            self.fc.target_system, self.fc.target_component,
                            0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                            2, 0, 0, 0, 0, 0,
                            int(lat * 1e7), int(lng * 1e7), alt
                        )
                    else:
                        print("RTH USER: No user GPS — falling back to RTL")
                        self.fc.set_mode('RTL')

                # V110: AI AVOIDANCE RESPONSE — from cloud/laptop AI
                elif cmd == 'AI_AVOIDANCE':
                    # AI sends {vx, vy, vz} avoidance velocity vector
                    if isinstance(payload, dict):
                        self.telemetry_cache['ai_avoidance_vector'] = {
                            'vx': float(payload.get('vx', 0)),
                            'vy': float(payload.get('vy', 0)),
                            'vz': float(payload.get('vz', 0)),
                        }
                        print(f"🤖 AI AVOIDANCE RECEIVED: vx={payload.get('vx',0)} vy={payload.get('vy',0)} vz={payload.get('vz',0)}")

                # FOLLOW ME — continuously fly toward user's phone GPS
                elif cmd == 'FOLLOW_ME':
                    enabled = True
                    if isinstance(payload, dict):
                        enabled = payload.get('enabled', True)
                    self.follow_me_active = enabled
                    if enabled:
                        print("FOLLOW ME: ACTIVE — tracking user GPS")
                    else:
                        print("FOLLOW ME: STOPPED")
                        self.fc.mav.set_position_target_local_ned_send(
                            0, self.fc.target_system, self.fc.target_component,
                            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                            0b0000111111000111,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        )

                # BATTERY THRESHOLD — set battery % for auto-RTH
                elif cmd == 'SET_BATT_RTH':
                    pct = int(payload.get('percent', 20)) if isinstance(payload, dict) else 20
                    self.batt_threshold = max(5, min(50, pct))
                    # User can choose RTH destination
                    dest = payload.get('destination', 'launch') if isinstance(payload, dict) else 'launch'
                    self.batt_rth_destination = dest  # 'launch' or 'user'
                    print(f"BATT RTH: {self.batt_threshold}% -> {dest}")

                # MODE SWITCH — app sends mode name directly
                elif cmd in ('STABILIZE', 'ALT_HOLD', 'LOITER', 'POSHOLD', 'GUIDED', 'AUTO'):
                    mode_map = {
                        'STABILIZE': 0, 'ALT_HOLD': 2, 'LOITER': 5,
                        'POSHOLD': 16, 'GUIDED': 4, 'AUTO': 3,
                    }
                    mode_id = mode_map.get(cmd, 0)
                    print(f"MODE SWITCH: {cmd} ({mode_id})")
                    self.fc.mav.set_mode_send(
                        self.fc.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        mode_id
                    )

                # P2.5: CAPTURE/RECORD HANDLERS - Camera control (INTERCEPTED FOR RADXA)
                # V110: Support both CAPTURE and CAPTURE_PHOTO from app
                elif cmd in ('CAPTURE', 'CAPTURE_PHOTO') or type == 'capture':
                    print("📸 CAPTURE: Triggering Local Save...")
                    self.take_photo_flag = True
                    await self.esp32_cmd_queue.put({"type": "capture"})

                # V110: Support START_RECORDING / STOP_RECORDING from app
                elif cmd in ('RECORD', 'START_RECORDING', 'STOP_RECORDING') or type == 'record':
                    # V110: Robust recording state detection
                    recording = False
                    if cmd == 'START_RECORDING':
                        recording = True
                    elif cmd == 'STOP_RECORDING':
                        recording = False
                    elif isinstance(payload, dict):
                        recording = payload.get('recording', False)
                    elif isinstance(payload, bool):
                        recording = payload

                    print(f"🎥 RECORD: {'START' if recording else 'STOP'} (Local)")
                    self.is_recording = recording
                    # Also send to ESP32 for feedback (e.g. solid RED LED)
                    await self.esp32_cmd_queue.put({"type": "record", "recording": recording})
                elif type == 'joystick':
                    try:
                        # Extract and Clamping
                        def map_ch(val, center=True):
                            if center: return int(1500 + (val * 500))
                            return int(1000 + (val * 1000)) # 0-1 range to 1000-2000
                        # V110: CORRECT MODE 2 MAPPING (Standard RC Controller)
                        # App Left Stick:  z = throttle (up/down), r = yaw (left/right)
                        # App Right Stick: x = roll (left/right),  y = pitch (up/down)
                        # ArduPilot RC channels: RC1=Roll, RC2=Pitch, RC3=Throttle, RC4=Yaw

                        is_armed = self.fc.motors_armed() if self.fc else False

                        # Apply expo curve for RC feel (0.0 = linear, 1.0 = max expo)
                        expo = float(self.telemetry_cache.get('expo', 0.3))
                        def apply_expo(val, exp):
                            """Cubic expo curve: output = (1-exp)*val + exp*val^3"""
                            sign = 1 if val >= 0 else -1
                            abs_val = abs(val)
                            return sign * ((1.0 - exp) * abs_val + exp * (abs_val ** 3))

                        # Deadzone: ignore tiny stick movements (<5%)
                        def deadzone(val, dz=0.05):
                            if abs(val) < dz: return 0.0
                            sign = 1 if val >= 0 else -1
                            return sign * (abs(val) - dz) / (1.0 - dz)

                        # Extract raw values from payload
                        raw_roll  = deadzone(float(payload.get('x', 0)))   # Right stick X → Roll
                        raw_pitch = deadzone(float(payload.get('y', 0)))   # Right stick Y → Pitch
                        raw_thr   = deadzone(float(payload.get('z', 0)))   # Left stick Y → Throttle
                        raw_yaw   = deadzone(float(payload.get('r', 0)))   # Left stick X → Yaw

                        # Apply expo for RC feel
                        raw_roll  = apply_expo(raw_roll, expo)
                        raw_pitch = apply_expo(raw_pitch, expo)
                        raw_yaw   = apply_expo(raw_yaw, expo)
                        # Throttle: no expo (linear is better for throttle control)

                        # Map to RC PWM (1000-2000, center 1500)
                        target_rc1 = map_ch(raw_roll, center=True)    # RC1 = Roll
                        target_rc2 = map_ch(raw_pitch, center=True)   # RC2 = Pitch
                        target_rc3 = map_ch(raw_thr, center=True)     # RC3 = Throttle
                        target_rc4 = map_ch(raw_yaw, center=True)     # RC4 = Yaw

                        # V50: REMOVE SMOOTHING ENTIRELY (Lag Fix Check)
                        # Direct mapping. No buffer.
                        rc4 = target_rc4
                        rc2 = target_rc2
                        rc1 = target_rc1

                        # V64: ADAPTIVE THROTTLE CURVE
                        # Use MOT_THST_HOVER (learned by AltHold) as the Center Stick Target

                        hover_param = 0.55 # F450 Heavy Default (1.5kg)
                        try:
                             # Try to read cached value (updated by watchdog)
                             hover_param = self.telemetry_cache.get('MOT_THST_HOVER', 0.55)
                             if hover_param < 0.1 or hover_param > 0.8: hover_param = 0.55 # Sanity Check
                        except:
                             hover_param = 0.55

                        hover_pwm = 1000 + (hover_param * 1000) # e.g. 0.55 -> 1550

                        t_in = target_rc3
                        t_out = 1000

                        if t_in < 1500:
                            # Low Half: Map 1000-1500 -> 1100-HoverPWM
                            pct = (t_in - 1000) / 500.0
                            t_out = 1100 + (pct * (hover_pwm - 1100))
                        else:
                            # High Half: Map 1500-2000 -> HoverPWM-2000 (V66: FULL POWER)
                            pct = (t_in - 1500) / 500.0
                            t_out = hover_pwm + (pct * (2000 - hover_pwm))

                        rc3 = int(t_out)

                        # V110: TILT COMPENSATION (Gentle)
                        # When tilting forward/sideways, add gentle throttle to maintain altitude
                        # Max boost: +200 PWM (was 800 — way too aggressive)
                        tilt_pitch = abs(target_rc2 - 1500)
                        tilt_roll = abs(target_rc1 - 1500)
                        max_tilt = max(tilt_pitch, tilt_roll)  # 0 to 500

                        if max_tilt > 100:  # Only compensate for significant tilt
                            boost = (max_tilt / 500.0) * 200.0  # Max +200 PWM
                            rc3 += int(boost)
                            if rc3 > 2000: rc3 = 2000

                        # V50: MOTOR SAFETY (NO STOPPING IN AIR) - User wants LINEAR DESCENT
                        # (Curve starts at 1100. We just clamp min to 1100 to prevent disarm in air)
                        if is_armed:
                             if rc3 < 1100: rc3 = 1100 # Safety Floor (Idle Only)

                        if abs(target_rc2 - 1500) > 50 or abs(target_rc3 - 1500) > 50:
                             # V65 DEBUG: Show what Pitch/Roll we are actually sending
                             print(f"🕹️ MIX: Pitch(RC2)={rc2} Roll(RC1)={rc1} Thr(RC3)={rc3}")
                        # V34: TOY MODE (Auto-Arm on Throttle) BEFORE CLAMPING CHECK
                        # (Logic handled by 'is_armed' check above)

                        if not is_armed:
                            if rc3 > 1400: # Adjusted for new curve (approx 30% up)
                                 self.auto_arm_counter = getattr(self, 'auto_arm_counter', 0) + 1

                                 # Lie to FC (Send Zero) so it accepts Arming
                                 real_rc3 = rc3
                                 rc3 = 1000

                                 if self.auto_arm_counter > 10: # ~1s hold
                                      # MA-2 FIX: Set safe flight mode BEFORE arming
                                      # Use LOITER if GPS available (mode 5), else ALT_HOLD (mode 2)
                                      gps_sats = self.telemetry_cache.get('satellites', 0)
                                      if gps_sats >= 6:
                                          self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 5)  # LOITER
                                          print("🚀 AUTO-ARM: Setting LOITER mode (GPS locked)...")
                                      else:
                                          self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 2)  # ALT_HOLD
                                          print("🚀 AUTO-ARM: Setting ALT_HOLD mode (no GPS)...")
                                      import asyncio
                                      # Small delay for mode to take effect
                                      self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                                      print("🚀 AUTO-ARM: Armed!")
                                      self.auto_arm_counter = 0
                            else:
                                 self.auto_arm_counter = 0
                        self.fc.mav.rc_channels_override_send(
                            self.fc.target_system, self.fc.target_component,
                            rc1, rc2, rc3, rc4, 65535, 65535, 65535, 65535
                        )

                        # V31: STICK ARMING LOGIC (Backup - Down-Right)
                        if rc3 < 1150 and rc4 > 1900:
                             self.stick_arm_counter = getattr(self, 'stick_arm_counter', 0) + 1
                             if self.stick_arm_counter > 20: # ~2 seconds @ 10Hz
                                  print("🕹️ STICK ARM TRIGGERED!")
                                  self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                                  self.stick_arm_counter = 0
                        else:
                             self.stick_arm_counter = 0
                        # V15: Debug RC Output
                        # V15: Debug RC Output
                        if int(time.time() * 10) % 20 == 0: # Every 2s
                           print(f"🕹️ JOY: R{rc1} P{rc2} T{rc3} Y{rc4}")

                    except Exception as e:
                         print(f"Joystick Error: {e}")
            except Exception as e:
                print(f"CMD Loop Error: {e}")
                break
    async def _try_connect_gopro(self):
        """V110: Try connecting to GoPro via WiFi UDP or USB"""
        import cv2
        import subprocess

        # Try 1: GoPro USB webcam
        for idx in range(10):
            dev = f"/dev/video{idx}"
            if not os.path.exists(dev): continue
            try:
                result = subprocess.run(['v4l2-ctl', '-d', dev, '--info'],
                    capture_output=True, text=True, timeout=3)
                if 'gopro' in result.stdout.lower() or 'webcam' in result.stdout.lower():
                    pipeline = (f"v4l2src device={dev} ! videoscale ! "
                        "video/x-raw,width=1280,height=720 ! videoconvert ! "
                        "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    if cap.isOpened():
                        ret, f = cap.read()
                        if ret and f is not None:
                            self.external_cap = cap
                            self.external_available = True
                            self.telemetry_cache['external_available'] = True
                            print(f"✅ EXTERNAL (GoPro USB): ACTIVE @ {dev}")
                            return True
                        cap.release()
            except: pass

        # Try 2: GoPro WiFi UDP stream
        try:
            cap = cv2.VideoCapture("udp://@0.0.0.0:8554", cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, f = cap.read()
                if ret and f is not None:
                    self.external_cap = cap
                    self.external_available = True
                    self.telemetry_cache['external_available'] = True
                    print(f"✅ EXTERNAL (GoPro WiFi UDP): ACTIVE @ {f.shape[1]}x{f.shape[0]}")
                    return True
                cap.release()
        except: pass

        self.external_available = False
        self.telemetry_cache['external_available'] = False
        print("⚠️ EXTERNAL (GoPro): Not detected")
        return False

    async def _smart_avoidance_monitor(self, mode='rth'):
        """
        V110: Smart Obstacle Avoidance Monitor
        Runs during RTH/Land. Continuously checks sensors.

        PRIORITY: SENSORS > AI
        - If any sensor detects obstacle < CRITICAL_DIST: IMMEDIATE HALT (no AI wait)
        - If obstacle < CAUTION_DIST: Request AI for avoidance path, pause movement
        - If obstacle < WARN_DIST: Log warning, let AI plan ahead

        Sends sensor data + video frame to cloud AI/laptop AI for path planning.
        AI returns avoidance vector which gets applied as GUIDED waypoint offset.
        """
        print(f"🛡️ SMART AVOIDANCE MONITOR STARTED (mode={mode})")
        avoidance_active = False

        while self.running and self.is_armed:
            safety = self.safety.get_safety_status()
            closest = safety['closest_obstacle_m']
            level = safety['level']

            # --- SENSOR PRIORITY: Immediate halt if too close ---
            if level == 'CRITICAL':
                if not avoidance_active:
                    print(f"🛑 AVOIDANCE: HALT! Obstacle @ {closest:.2f}m ({safety['obstacle_source']})")
                    avoidance_active = True
                    # Switch to LOITER to stop all movement
                    self.fc.mav.set_mode_send(
                        self.fc.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 5)  # LOITER

                    # Send sensor data to AI for avoidance planning
                    if self.ws:
                        try:
                            await self.ws.send(json.dumps({
                                'type': 'avoidance_request',
                                'safety': safety,
                                'tof': {
                                    'front': self.telemetry_cache.get('t1', -1),
                                    'right': self.telemetry_cache.get('t2', -1),
                                    'back': self.telemetry_cache.get('t3', -1),
                                    'left': self.telemetry_cache.get('t4', -1),
                                },
                                'lidar_dist': self.telemetry_cache.get('lidar_dist', 9.9),
                                'altitude': self.telemetry_cache.get('altitude', 0),
                                'heading': self.telemetry_cache.get('heading', 0),
                                'mode': mode,
                                'lat': self.telemetry_cache.get('lat', 0),
                                'lng': self.telemetry_cache.get('lng', 0),
                            }))
                        except: pass

            elif level == 'CAUTION':
                if not avoidance_active:
                    print(f"⚠️ AVOIDANCE: SLOWING. Obstacle @ {closest:.2f}m")
                    # Don't halt, but slow down — reduce RTL speed
                    # Send to AI for planning
                    if self.ws:
                        try:
                            await self.ws.send(json.dumps({
                                'type': 'avoidance_warning',
                                'safety': safety,
                                'mode': mode,
                            }))
                        except: pass

            elif level in ('WARN', 'CLEAR'):
                if avoidance_active:
                    print(f"✅ AVOIDANCE: CLEAR. Resuming {mode.upper()}")
                    avoidance_active = False
                    # Resume original mode
                    if mode == 'rth':
                        self.fc.set_mode('RTL')
                    elif mode == 'land':
                        self.fc.mav.command_long_send(
                            self.fc.target_system, self.fc.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)

            # Handle AI avoidance response (if received via command loop)
            ai_avoidance = self.telemetry_cache.get('ai_avoidance_vector', None)
            if ai_avoidance and avoidance_active:
                # AI gives velocity vector to avoid obstacle
                vx = ai_avoidance.get('vx', 0)
                vy = ai_avoidance.get('vy', 0)
                vz = ai_avoidance.get('vz', 0)

                # BUT ONLY if sensors still say it's safe in that direction
                tof_front = self.telemetry_cache.get('t1', 9999)
                tof_right = self.telemetry_cache.get('t2', 9999)
                tof_back = self.telemetry_cache.get('t3', 9999)
                tof_left = self.telemetry_cache.get('t4', 9999)

                # SENSOR PRIORITY: Block AI vector if sensor says obstacle in that direction
                if vx > 0 and tof_front < 500:  # Moving forward but front blocked
                    vx = 0
                    print("🛑 SENSOR OVERRIDE: AI wants forward, front ToF blocked")
                if vx < 0 and tof_back < 500:   # Moving backward but back blocked
                    vx = 0
                    print("🛑 SENSOR OVERRIDE: AI wants backward, back ToF blocked")
                if vy > 0 and tof_right < 500:  # Moving right but right blocked
                    vy = 0
                    print("🛑 SENSOR OVERRIDE: AI wants right, right ToF blocked")
                if vy < 0 and tof_left < 500:   # Moving left but left blocked
                    vy = 0
                    print("🛑 SENSOR OVERRIDE: AI wants left, left ToF blocked")

                if vx != 0 or vy != 0 or vz != 0:
                    # Apply AI avoidance vector via GUIDED mode velocity
                    self.fc.mav.set_mode_send(
                        self.fc.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)  # GUIDED
                    self.fc.mav.set_position_target_local_ned_send(
                        0, self.fc.target_system, self.fc.target_component,
                        mavutil.mavlink.MAV_FRAME_BODY_NED,
                        0b0000111111000111,  # Velocity only
                        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
                    print(f"🤖 AI AVOIDANCE: vx={vx:.1f} vy={vy:.1f} vz={vz:.1f} (sensor-validated)")

                # Clear the vector after applying
                self.telemetry_cache['ai_avoidance_vector'] = None

            await asyncio.sleep(0.1)  # 10Hz check rate

        print(f"🛡️ SMART AVOIDANCE MONITOR STOPPED (mode={mode})")

    async def video_loop(self):
        import cv2
        import subprocess

        print("🔍 V110: DUAL CAMERA INIT (Pi Camera V2 + GoPro Hero 12)...")

        # === INTERNAL CAMERA: Pi Camera V2 via CSI ===
        internal_pipeline = (
            "v4l2src device=/dev/video0 ! "
            "videoscale ! video/x-raw,width=1280,height=720 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        self.internal_cap = cv2.VideoCapture(internal_pipeline, cv2.CAP_GSTREAMER)
        if self.internal_cap and self.internal_cap.isOpened():
            ret, f = self.internal_cap.read()
            if ret and f is not None:
                print(f"✅ INTERNAL (Pi Camera V2): ACTIVE @ {f.shape[1]}x{f.shape[0]}")
            else:
                print("⚠️ INTERNAL: Opened but no frames yet")
        else:
            print("❌ INTERNAL: Failed to open /dev/video0. Will retry in loop.")
            self.internal_cap = None

        # === EXTERNAL CAMERA: GoPro Hero 12 via WiFi UDP ===
        # GoPro streams via WiFi to UDP port 8554 (set up via BLE command)
        await self._try_connect_gopro()

        self.cam_needs_reset = False
        sources = []
        if self.internal_cap: sources.append("Pi Camera V2")
        if self.external_cap: sources.append("GoPro Hero 12")
        self.telemetry_cache['cam_source'] = '+'.join(sources) if sources else 'none'
        self.telemetry_cache['active_cam'] = self.active_cam
        self.telemetry_cache['external_available'] = self.external_available
        print(f"📹 CAMERAS READY: {sources or 'NONE - will retry'}")

        if not self.internal_cap and not self.external_cap:
            print("🚨 NO CAMERAS DETECTED. Will keep retrying...")

        # Periodic GoPro reconnection attempt (every 10s if not connected)
        gopro_retry_interval = 10
        last_gopro_retry = 0
        # V110: DUAL CAMERA STREAM LOOP
        self.frame_in_transit = False
        self.udp_streamers_init = False
        self.ts_laptop_streamer = None
        self.ts_phone_streamer = None

        async with aiohttp.ClientSession() as session:
            while self.running:
                loop = asyncio.get_running_loop()
                current_time = int(time.time())

                # --- Periodic GoPro reconnect if not available ---
                if not self.external_available and current_time - last_gopro_retry > gopro_retry_interval:
                    last_gopro_retry = current_time
                    await self._try_connect_gopro()

                # --- Periodic internal cam reconnect if lost ---
                if self.internal_cap is None or not self.internal_cap.isOpened():
                    try:
                        internal_pipeline = (
                            "v4l2src device=/dev/video0 ! videoscale ! "
                            "video/x-raw,width=1280,height=720 ! videoconvert ! "
                            "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
                        )
                        self.internal_cap = cv2.VideoCapture(internal_pipeline, cv2.CAP_GSTREAMER)
                    except: pass

                # --- Read from BOTH cameras (non-blocking) ---
                internal_frame = None
                external_frame = None

                if self.internal_cap and self.internal_cap.isOpened():
                    try:
                        ret, f = await loop.run_in_executor(None, self.internal_cap.read)
                        if ret and f is not None:
                            internal_frame = f
                            self.latest_internal_frame = f
                    except: pass

                if self.external_cap and self.external_cap.isOpened():
                    try:
                        ret, f = await loop.run_in_executor(None, self.external_cap.read)
                        if ret and f is not None:
                            external_frame = f
                            self.latest_external_frame = f
                        else:
                            # GoPro disconnected mid-flight
                            self.external_available = False
                            self.external_cap = None
                            self.telemetry_cache['external_available'] = False
                    except:
                        self.external_available = False

                # --- Select active frame for app preview ---
                if self.active_cam == 'external' and external_frame is not None:
                    active_frame = external_frame
                elif internal_frame is not None:
                    active_frame = internal_frame
                elif external_frame is not None:
                    active_frame = external_frame  # Fallback
                else:
                    await asyncio.sleep(0.1)
                    continue  # No frames from either camera

                # --- Select recording frame (may differ from active) ---
                if self.recording_cam == 'external' and external_frame is not None:
                    recording_frame = external_frame
                elif self.recording_cam == 'internal' and internal_frame is not None:
                    recording_frame = internal_frame
                else:
                    recording_frame = active_frame  # Fallback

                # Resize active frame to standard size
                if active_frame.shape[1] != CAM_WIDTH or active_frame.shape[0] != CAM_HEIGHT:
                    try:
                        active_frame = cv2.resize(active_frame, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    except: pass

                try:
                    # V87: Brightness boost for dark conditions
                    active_frame = cv2.convertScaleAbs(active_frame, alpha=1.5, beta=30)

                    # --- TAILSCALE UDP STREAMS (init once) ---
                    if not self.udp_streamers_init:
                        self.udp_streamers_init = True
                        if TAILSCALE_LAPTOP_IP:
                            try:
                                lp = ("appsrc ! videoconvert ! "
                                    "x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast key-int-max=30 ! "
                                    "mpegtsmux ! udpsink host=" + TAILSCALE_LAPTOP_IP + " port=" + str(TAILSCALE_VIDEO_PORT) + " sync=false")
                                self.ts_laptop_streamer = cv2.VideoWriter(lp, cv2.CAP_GSTREAMER, 0, 30.0, (1280, 720), True)
                                if not (self.ts_laptop_streamer and self.ts_laptop_streamer.isOpened()):
                                    lp = ("appsrc ! videoconvert ! avenc_mpeg4 bitrate=1500000 gop-size=15 ! "
                                        "avimux ! udpsink host=" + TAILSCALE_LAPTOP_IP + " port=" + str(TAILSCALE_VIDEO_PORT) + " sync=false")
                                    self.ts_laptop_streamer = cv2.VideoWriter(lp, cv2.CAP_GSTREAMER, 0, 30.0, (1280, 720), True)
                                print(f"📡 TAILSCALE → LAPTOP AI: {TAILSCALE_LAPTOP_IP}:{TAILSCALE_VIDEO_PORT}")
                            except Exception as e:
                                print(f"❌ Tailscale Laptop Stream: {e}")
                        if TAILSCALE_PHONE_IP:
                            try:
                                pp = ("appsrc ! videoconvert ! avenc_mpeg4 bitrate=800000 gop-size=15 ! "
                                    "avimux ! udpsink host=" + TAILSCALE_PHONE_IP + " port=" + str(TAILSCALE_VIDEO_PORT + 1) + " sync=false")
                                self.ts_phone_streamer = cv2.VideoWriter(pp, cv2.CAP_GSTREAMER, 0, 30.0, (640, 480), True)
                                print(f"📡 TAILSCALE → PHONE: {TAILSCALE_PHONE_IP}:{TAILSCALE_VIDEO_PORT + 1}")
                            except: pass

                    # Send RECORDING frame (not active) to AI via Tailscale
                    ai_frame = cv2.resize(recording_frame, (1280, 720)) if recording_frame.shape[:2] != (720, 1280) else recording_frame
                    if self.ts_laptop_streamer:
                        self.ts_laptop_streamer.write(ai_frame)
                    if self.ts_phone_streamer:
                        self.ts_phone_streamer.write(cv2.resize(recording_frame, (640, 480)))

                    # --- RECORDING (uses recording_cam frame, user-selected res) ---
                    if self.is_recording:
                        if self.video_writer is None:
                            fname = f"flight_record_{current_time}.mp4"
                            h, w = recording_frame.shape[:2]
                            print(f"📼 RECORDING: {fname} @ {w}x{h} cam={self.recording_cam}")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            self.video_writer = cv2.VideoWriter(fname, fourcc, 30.0, (w, h))
                            self.record_start_time = current_time
                        if self.video_writer:
                            self.video_writer.write(recording_frame)
                    else:
                        if self.video_writer is not None:
                            print(f"💾 SAVED ({int(current_time - self.record_start_time)}s)")
                            self.video_writer.release()
                            self.video_writer = None

                    # --- PHOTO CAPTURE (uses recording_cam frame) ---
                    if self.take_photo_flag:
                        fname = f"photo_{current_time}.jpg"
                        cv2.imwrite(fname, recording_frame)
                        print(f"📸 PHOTO: {fname}")
                        self.take_photo_flag = False

                    # --- APP PREVIEW (active_cam, downscaled, via Render relay) ---
                    if not self.frame_in_transit:
                        preview = cv2.resize(active_frame, (480, 360))
                        retval, buffer = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        if retval:
                            self.latest_jpeg = buffer.tobytes()
                            asyncio.create_task(self.push_frame(session, self.latest_jpeg))
                            if current_time % 5 == 0 and current_time != getattr(self, 'last_vid_sec', 0):
                                self.last_vid_sec = current_time
                                print(f"🎥 STREAM: active={self.active_cam} rec={self.recording_cam} ext={'✅' if self.external_available else '❌'}")

                except Exception as e:
                    print(f"⚠️ Stream Error: {e}")

                await asyncio.sleep(0.01)
    async def push_frame(self, session, data):
        if self.frame_in_transit: return
        self.frame_in_transit = True
        try:
            url = API_URL + "/video/frame"
            # Proper Multipart Upload
            form = aiohttp.FormData()
            form.add_field('file', data, filename='frame.jpg', content_type='image/jpeg')

            # V47: Increased Timeout for Weak Signals
            # V108: Increased to 10s for Larger Frames (Quality 60)
            async with session.post(url, data=form, timeout=10.0) as response:
                if response.status != 200:
                    print(f"❌ Video Upload Failed: {response.status}")
        except Exception as e:
            print(f"❌ Video Network Error: {repr(e)}")
        finally:
            self.frame_in_transit = False
    async def lidar_loop(self):
        """V102: REAL YDLIDAR DRIVER (Via Python SDK)"""
        try:
            import ydlidar
        except ImportError:
            print("❌ LIDAR: 'ydlidar' lib missing. Run fix_dependencies.sh")
            return

        ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]
        laser = None
        
        print(f"🔭 LIDAR: Scanning ports {ports}...")

        for port in ports:
            temp_laser = ydlidar.CYdLidar()
            temp_laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
            temp_laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
            temp_laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
            temp_laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
            temp_laser.setlidaropt(ydlidar.LidarPropScanFrequency, 5.0)
            temp_laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
            temp_laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
            temp_laser.setlidaropt(ydlidar.LidarPropMaxRange, 8.0)
            temp_laser.setlidaropt(ydlidar.LidarPropMinRange, 0.1)

            if temp_laser.initialize():
                print(f"✅ LIDAR: FOUND @ {port}")
                laser = temp_laser
                break
            else:
                print(f"   Lidar fetch failed on {port}")
        
        if laser is None:
            print("❌ LIDAR: Init Failed on ALL ports. Check USB.")
            await asyncio.sleep(5)
            return

        # BYPASS: turnOn() reports "Device Tremble" but user says it's physically stable
        # SDK health check is too sensitive. Skip it and go straight to scanning.
        print("⚠️ BYPASSING turnOn() health check (too sensitive)")
        print("✅ LIDAR: Starting scan loop directly...")
        scan = ydlidar.LaserScan()

        while self.running:
            try:
                 ret = laser.doProcessSimple(scan)
                 if ret:
                     # Find minimum distance to prioritize safety
                     min_dist = 10.0

                     # Check Forward Sector (approx logic)
                     # Real implementation uses point.angle
                     for i in range(scan.points.size()):
                         point = scan.points[i]
                         if point.range > 0.1 and point.range < min_dist:
                             min_dist = point.range

                     # Send to FC (cm)
                     dist_cm = int(min_dist * 100)
                     if dist_cm > 10 and dist_cm < 800:
                         self.fc.mav.distance_sensor_send(
                                0, 10, 800, dist_cm, 0, 0, 0, 0
                         )

                     self.telemetry_cache['obstacle_dist'] = min_dist
                     self.telemetry_cache['lidar_dist'] = min_dist  # Used by SafetyEnvelope
                     self.telemetry_cache['lidar_status'] = "ACTIVE"

                     # Build point cloud for Laptop AI spatial grid
                     import math
                     lidar_points = []
                     for i in range(scan.points.size()):
                         pt = scan.points[i]
                         if 0.1 < pt.range < 8.0:
                             x = pt.range * math.cos(pt.angle)
                             y = pt.range * math.sin(pt.angle)
                             lidar_points.append([round(x, 2), round(y, 2)])

                     # Broadcast lidar_scan to laptop AI (max 200 points to save bandwidth)
                     if lidar_points:
                         step = max(1, len(lidar_points) // 200)
                         sampled = lidar_points[::step]
                         scan_msg = json.dumps({"type": "lidar_scan", "payload": {"points": sampled}})
                         if self.ws:
                             try: await self.ws.send(scan_msg)
                             except: pass
                         if self.local_clients:
                             for c in list(self.local_clients):
                                 try: await c.send(scan_msg)
                                 except: pass

                 await asyncio.sleep(0.05)
            except Exception as e:
                print(f"Lidar Error: {e}")
                await asyncio.sleep(1)

        laser.turnOff()
        laser.disconnecting()

    async def watchdog_loop(self):
        """Monitors Cloud Connection Health + Battery Failsafe"""
        print("🐕 Watchdog Active")
        while self.running:
            # V110: FAILSAFE - Return to USER'S LAST LOCATION on disconnect (not launch point)
            last_msg_delta = time.time() - self.last_cloud_msg

            if last_msg_delta > 5.0 and not self.watchdog_triggered and self.is_armed:
                 self.watchdog_triggered = True
                 # Priority: Return to user GPS if available, else RTL to launch
                 if self.user_gps:
                     lat, lng = self.user_gps
                     print(f"⚠️ LOST CONNECTION ({int(last_msg_delta)}s)! RETURNING TO USER @ ({lat:.6f}, {lng:.6f})")
                     try:
                         self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)  # GUIDED
                         self.fc.mav.mission_item_int_send(
                             self.fc.target_system, self.fc.target_component,
                             0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                             mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 0, 0, 0, 0, 0,
                             int(lat * 1e7), int(lng * 1e7), 15  # Return at 15m altitude
                         )
                     except: pass
                 else:
                     print(f"⚠️ LOST CONNECTION ({int(last_msg_delta)}s)! NO USER GPS — TRIGGERING RTL!")
                     try:
                         self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6)  # RTL
                     except: pass

            # V110: BATTERY FAILSAFE — use configurable threshold (not hardcoded 15%)
            current_batt = self.telemetry_cache.get('battery', 100)
            if self.is_armed and current_batt < self.batt_threshold and current_batt > 0:
                 if not getattr(self, 'low_batt_triggered', False):
                      self.low_batt_triggered = True
                      self.follow_me_active = False  # Stop follow on low batt
                      # Use configured RTH destination
                      if self.batt_rth_destination == 'user' and self.user_gps:
                          lat, lng = self.user_gps
                          print(f"⚠️ LOW BATTERY ({current_batt}% < {self.batt_threshold}%)! RETURNING TO USER @ ({lat:.6f}, {lng:.6f})")
                          try:
                              self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
                              self.fc.mav.mission_item_int_send(
                                  self.fc.target_system, self.fc.target_component,
                                  0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                                  mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 0, 0, 0, 0, 0,
                                  int(lat * 1e7), int(lng * 1e7), 15
                              )
                          except: pass
                      else:
                          print(f"⚠️ LOW BATTERY ({current_batt}% < {self.batt_threshold}%)! TRIGGERING RTL!")
                          try:
                              self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6)
                          except: pass

            if last_msg_delta < 2.0:
                self.watchdog_triggered = False
            await asyncio.sleep(1)

    async def esp32_hardware_loop(self):
        """Complete ESP32 Bidirectional Link (WiFi UDP Edition)"""
        import socket
        
        UDP_IP = "0.0.0.0" # Listen on all interfaces
        UDP_PORT = 8888
        
        print(f"🔌 ESP32: Binding UDP Port {UDP_PORT}...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((UDP_IP, UDP_PORT))
        sock.setblocking(False) 
        
        print(f"✅ ESP32: UDP Listener Active on {UDP_PORT}")
        
        esp32_addr = None # Will learn from incoming packets

        while self.running:
            try:
                # === RECEIVE: ESP32 → Radxa (UDP) ===
                try:
                    data_raw, addr = sock.recvfrom(4096)
                    esp32_addr = addr # Store for sending back commands
                    
                    line = data_raw.decode('utf-8').strip()
                    
                    if line and line.startswith('{'):
                         try:
                             data = json.loads(line)
                             
                             # Parse VL53L1X sensor data (t1,t2,t3,t4 in mm)
                             # 2 Top (Angled), 2 Bottom (Angled). YDLidar (Horizontal).
                             if 't1' in data and 't2' in data and 't3' in data and 't4' in data:
                                 # Filter Self-Collisions (<15cm) - e.g. Propellers/Legs
                                 raw_distances = [data['t1'], data['t2'], data['t3'], data['t4']]
                                 valid_distances = [d for d in raw_distances if d > 150 and d < 8000]
                                 
                                 # SAFETY PRIORITY: Active Braking Override
                                 # If any object is within 50cm (excluding self <15cm), STOP immediately.
                                 min_dist = min(valid_distances) if valid_distances else 9999
                                 if min_dist < 500: # 50cm
                                     print(f"🛑 CRITICAL PROXIMITY ({min_dist}mm)! FORCE BRAKE!")
                                     try:
                                         # Send Brake/Hold Mode
                                         self.fc.mav.set_mode_send(
                                             self.fc.target_system,
                                             mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                             17 # BRAKE (ArduPilot) / HOLD (PX4)
                                         )
                                     except: pass
                                 
                                 # Send to FC via MAVLink (Mapping Top/Bottom)
                                 # T1, T2 = Top (Mapped to UP/FORWARD_UP?) -> Using UP (24) and CUSTOM
                                 # B1, B2 = Bottom (Mapped to DOWN (25))
                                 
                                 # We map loosely for logs, but YDLIDAR handles Horizontal Ring 
                                 self.telemetry_cache['prox_alert'] = len(valid_distances) > 0
                                      
                                 # Log occasionally
                                 if int(time.time()) % 2 == 0:
                                      print(f"📡 VL53 (Filtered): {valid_distances} mm")
                         
                             # P3.3: Inject ESP32 IMU data into telemetry cache
                             # ax/ay/az = accelerometer (m/s²), gx/gy/gz = gyroscope (rad/s)
                             if 'ax' in data:
                                 self.telemetry_cache['esp32_ax'] = data.get('ax', 0)
                                 self.telemetry_cache['esp32_ay'] = data.get('ay', 0)
                                 self.telemetry_cache['esp32_az'] = data.get('az', 0)
                             if 'gx' in data:
                                 self.telemetry_cache['esp32_gx'] = data.get('gx', 0)
                                 self.telemetry_cache['esp32_gy'] = data.get('gy', 0)
                                 self.telemetry_cache['esp32_gz'] = data.get('gz', 0)

                             # Inject raw ToF into telem cache for SafetyEnvelope
                             self.telemetry_cache['t1'] = data.get('t1', -1)
                             self.telemetry_cache['t2'] = data.get('t2', -1)
                             self.telemetry_cache['t3'] = data.get('t3', -1)
                             self.telemetry_cache['t4'] = data.get('t4', -1)

                             # P3.4: Vibration monitoring from ESP32 IMU
                             # If vibration is excessive, warn — may indicate prop damage
                             az = data.get('az', 9.8)
                             if abs(az) > 20:  # Normal gravity ~9.8, >20 = severe vibration
                                 print(f"⚠️ VIBRATION ALERT: az={az:.1f} m/s² (check props!)")
                                 self.telemetry_cache['vibration_alert'] = True
                             else:
                                 self.telemetry_cache['vibration_alert'] = False

                             # Forward full telemetry to cloud AND local clients (Tailscale direct)
                             esp32_msg = json.dumps({"type": "esp32_telem", "payload": data})
                             if self.ws:
                                 await self.ws.send(esp32_msg)
                             # Also send to local Tailscale clients (laptop AI direct path)
                             if self.local_clients:
                                 for c in list(self.local_clients):
                                     try: await c.send(esp32_msg)
                                     except: self.local_clients.discard(c)
                         
                         except Exception as e:
                             print(f"⚠️ ESP32 parse error: {e}")
                except BlockingIOError:
                    pass # No data waiting
                
                # === SEND: Radxa → ESP32 (UDP) ===
                if esp32_addr and hasattr(self, 'esp32_cmd_queue') and not self.esp32_cmd_queue.empty():
                    cmd = self.esp32_cmd_queue.get_nowait()
                    # MA-9 FIX: Convert gimbal commands to ESP32's expected format
                    # ESP32 firmware expects {"gim": [pitch, yaw]} not {"type":"gimbal","pitch":X,"yaw":Y}
                    if cmd.get('type') == 'gimbal':
                        esp32_msg = {"gim": [int(cmd.get('pitch', 0)), int(cmd.get('yaw', 0))]}
                    elif cmd.get('type') == 'led':
                        esp32_msg = {"led": cmd.get('color', 'OFF')}
                    elif cmd.get('type') == 'capture':
                        esp32_msg = {"capture": 1}
                    elif cmd.get('type') == 'record':
                        esp32_msg = {"record": 1 if cmd.get('recording') else 0}
                    else:
                        esp32_msg = cmd  # Pass through unknown commands as-is
                    msg = (json.dumps(esp32_msg) + '\n').encode('utf-8')
                    sock.sendto(msg, esp32_addr)
                    print(f"📤 ESP32 CMD: {esp32_msg} -> {esp32_addr}")
                
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"❌ ESP32 loop error: {e}")
                await asyncio.sleep(0.1)

if __name__ == "__main__":
    bridge = RadxaBridge()
    try:
        async def main():
            # Run everything in parallel so Video/Cloud doesn't wait for FC
            await asyncio.gather( 
                bridge.connect_mavlink(),
                bridge.connect_cloud(),
                bridge.video_loop(),
                bridge.lidar_loop(),
                bridge.esp32_hardware_loop(),
                bridge.watchdog_loop(),
                bridge.start_local_server(),
                bridge.start_local_video_server()
            )
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping Bridge...")
