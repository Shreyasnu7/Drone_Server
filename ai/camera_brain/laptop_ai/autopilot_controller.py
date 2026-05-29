
import asyncio
import time
from pymavlink import mavutil

class AutopilotController:
    """
    Connects to the Flight Controller (FC) via MAVLink.
    Executed by DirectorCore to move the physical drone.
    
    Port: /dev/ttyS0 (Radxa UART) or /dev/ttyACM0 (USB)
    """
    def __init__(self, connection_string=None, baud=57600, messaging_client=None):
        self.messaging = messaging_client
        # Auto-detect or use provided
        if connection_string:
            self.conn_str = connection_string
        else:
            # Try default UART for Radxa Zero 3W (Pins 8/10 usually ttyS0 or ttyAML0)
            # Fallback to USB if available
            import os
            # REMOTE MODE (Via Server): We don't connect to local serial ports.
            # Only connect if explicitly requested or on Linux/Radxa.
            if os.name == 'nt':
                print("💻 ON WINDOWS (Laptop AI Mode): Skipping local MAVLink connection. Commands will go via Server.")
                self.conn_str = None
            elif os.path.exists("/dev/ttyACM0"):
                self.conn_str = "/dev/ttyACM0"
            elif os.path.exists("/dev/ttyS0"):
                self.conn_str = "/dev/ttyS0" # UART
            else:
                self.conn_str = "/dev/ttyACM0" # Fallback
                
        self.baud = baud
        self.master = None
        self.baud = baud
        self.master = None
        self.connected = False
        
    def connect(self):
        # SKIP if no connection string (Remote Mode)
        if not self.conn_str:
            print("💻 Autopilot in REMOTE MODE (Relaying commands via Server)")
            if self.messaging:
                self.messaging.add_recv_handler(self._handle_telemetry)
            self.connected = True # Virtual connection to server
            return

        try:
            print(f"🔌 Connecting to Flight Controller on {self.conn_str}...")
            self.master = mavutil.mavlink_connection(self.conn_str, baud=self.baud)
            self.master.wait_heartbeat(timeout=5)
            self.connected = True
            print("✅ Flight Controller CONNECTED via MAVLink!")
        except Exception as e:
            print(f"⚠️ MAVLink Connection Failed: {e}")

    async def _handle_telemetry(self, packet):
        """ Capture telemetry from Server -> Drone """
        if packet.get("type") == "telemetry":
            # Bridge sends under "payload", older code used "data" — handle both
            self.latest_telem = packet.get("payload", packet.get("data", {}))
        elif packet.get("type") == "esp32_telem":
            # ESP32 sensor data — store for DirectorCore to read
            self.latest_esp32 = packet.get("payload", {})
    def set_mode(self, mode):
        if not self.master: return
        mode_id = self.master.mode_mapping().get(mode)
        if mode_id is None:
            print(f"Unknown mode: {mode}")
            return
        
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print(f"Requested Mode: {mode}")

    def arm(self):
        if not self.master: return
        self.master.arducopter_arm()
        self.master.motors_armed_wait()
        print("✅ Drone ARMED")

    def disarm(self):
        if not self.master: return
        self.master.arducopter_disarm()
        self.master.motors_disarmed_wait()
        print("⚠️ Drone DISARMED")

    def send_velocity(self, vx, vy, vz, yaw_rate=0):
        """
        Send LOCAL frame velocity commands (NED convention).
        Input: meters/second
        """
        if not self.connected or not self.master:
            # REMOTE MODE: Send to Server via Websocket
            if self.messaging:
                cmd = {
                    "type": "cmd_vel",
                    "vx": vx, "vy": vy, "vz": vz, "yaw_rate": yaw_rate,
                    "target": "drone" # Forward to drone
                }
                asyncio.create_task(self.messaging.send(cmd))
            return 
            
        # Create SET_POSITION_TARGET_LOCAL_NED message
        # type_mask: ignore pos, accel, only accept vel
        type_mask = 0b0000111111000111
        
        self.master.mav.set_position_target_local_ned_send(
            0, # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0, 0, 0, # x, y, z positions (ignored)
            vx, vy, vz, # velocities
            0, 0, 0, # accel (ignored)
            0, # yaw (ignored)
            yaw_rate # yaw_rate
        )

    def return_to_launch(self):
        if not self.master: return
        print("🏠 Returning to Launch (RTL)...")
        # Ensure mode is GUIDED or RTL
        self.set_mode("RTL") 
        # Alternatively use Command
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    def takeoff(self, altitude=2.0):
        if not self.master: return
        print(f"🛫 Taking off to {altitude}m...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )

    def fly_to_coords(self, lat, lon, alt=5.0):
        """
        Fly to specific Global Coordinates (Guided Mode).
        lat/lon in degrees (float). alt in meters (relative to home).
        """
        if not self.master: return
        print(f"📍 Flying to {lat}, {lon} at {alt}m")
        self.set_mode("GUIDED")
        
        # MAVLink requires integers for lat/lon (deg * 1e7)
        lat_int = int(lat * 1e7)
        lon_int = int(lon * 1e7)
        
        type_mask = 0b0000111111111000 # Ignore velocity/accel/yaw, only use Position
        
        self.master.mav.set_position_target_global_int_send(
            0, # boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            type_mask,
            lat_int, lon_int, alt,
            0, 0, 0, # vel
            0, 0, 0, # accel
            0, 0 # yaw
        )

    def execute_primitive(self, primitive):
        """
        Executes a high-level primitive locally.
        """
        if not self.connected: return
        
        action = primitive.get("action", "HOVER").upper()
        params = primitive.get("params", {})
        
        print(f"✈️ Autopilot Executing: {action}")
        
        if action == "TAKEOFF":
            self.set_mode("GUIDED")
            self.arm()
            time.sleep(1)
            self.takeoff(3.0) # Default altitude

        elif action == "HOVER":
            self.send_velocity(0, 0, 0)
            
        elif action == "FOLLOW":
            # Simple Forward velocity (0.5 m/s) with Yaw tracking (Not implemented here, needs Vision Loop)
            # This is a 'Kickstart' command. The vision loop should perform the actual tracking.
            self.send_velocity(0.5, 0, 0)
            
        elif action == "ORBIT":
            # Constant Yaw Rate, minor sideways velocity
            self.send_velocity(0, 0.5, 0, yaw_rate=0.2)
            
        elif action == "DRONIE":
            # Backwards and Upwards (Selfie Shot)
            # Body Frame: vx negative (back), vz negative (up)
            print("🚀 EXECUTING DRONIE SHOT")
            self.send_velocity(-1.5, 0, -1.0)

        elif action == "LAND":
            # MAVLink Land
            if self.master:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_LAND,
                    0, 0, 0, 0, 0, 0, 0, 0
                )

    def execute_plan_point(self, position, yaw):
        """
        Executes a 3D point from the UltraDirector curve.
        """
        # For simplicity in this v1, we map position changes to velocity
        pass

    def set_gimbal(self, pitch, yaw, roll=0):
        """
        Control Gimbal/Mount via MAVLink (MAV_CMD_DO_MOUNT_CONTROL).
        pitch, yaw, roll in Degrees.
        """
        if not self.master:
             # Remote Mode: Send to Server
             if self.messaging:
                 cmd = {
                     "type": "cmd_mount",
                     "pitch": pitch, "yaw": yaw, "roll": roll,
                     "target": "drone"
                 }
                 asyncio.create_task(self.messaging.send(cmd))
             return

        print(f"🔭 GIMBAL COMMAND: Pitch={pitch} Yaw={yaw}")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
            0,
            pitch, # param1: pitch (deg * 100 if extended, but standard is deg)
            roll,  # param2: roll
            yaw,   # param3: yaw
            0, 0, 0,
            mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING
        )

    def get_telemetry(self):
        """
        Returns telemetry (heading, battery).
        """
        if hasattr(self, 'latest_telem') and self.latest_telem:
            return self.latest_telem
        return {"heading": 0.0, "battery": 100}

    def get_position(self):
        """
        Returns [lat, lon, alt].
        """
        if hasattr(self, 'latest_telem') and self.latest_telem:
            gps = self.latest_telem.get("gps_data", {})
            return [gps.get("lat", 0), gps.get("lon", 0), gps.get("alt", 0)]
        return [0.0, 0.0, 0.0]
