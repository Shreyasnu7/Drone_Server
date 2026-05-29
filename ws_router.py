# ws_router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import os
from typing import Dict, List
from cloud_ai.dependencies import get_orchestrator

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        # Store active connections — SEPARATE slots for drone and laptop
        self.mobile_clients: List[WebSocket] = []
        self.drone_client: WebSocket | None = None    # RADXA_X / Cubie A7Z (physical drone)
        self.laptop_client: WebSocket | None = None    # laptop_vision (DirectorCore AI)

    async def connect_mobile(self, websocket: WebSocket):
        await websocket.accept()
        self.mobile_clients.append(websocket)
        print(f"📱 Mobile Client Connected. Total: {len(self.mobile_clients)}")

    async def connect_drone(self, websocket: WebSocket, client_id: str = ""):
        await websocket.accept()
        if client_id == "laptop_vision":
            self.laptop_client = websocket
            print("💻 Laptop AI Connected!")
        else:
            self.drone_client = websocket
            print("🚁 Drone (Radxa/Cubie) Connected!")

    def disconnect_mobile(self, websocket: WebSocket):
        if websocket in self.mobile_clients:
            self.mobile_clients.remove(websocket)
            print("📱 Mobile Client Disconnected")

    def disconnect_drone(self, websocket: WebSocket = None):
        if websocket == self.laptop_client:
            self.laptop_client = None
            print("💻 Laptop AI Disconnected")
        else:
            self.drone_client = None
            print("🚁 Drone Disconnected")

    async def broadcast_to_mobile(self, message: str):
        """Relay message from Drone to ALL Mobile Clients"""
        for connection in self.mobile_clients:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to mobile: {e}")

    async def send_to_drone(self, message: str):
        """Relay message from Mobile/Laptop to Drone"""
        if self.drone_client:
            try:
                await self.drone_client.send_text(message)
            except Exception as e:
                print(f"Error sending to drone: {e}")

    async def send_to_laptop(self, message: str):
        """Relay message from Drone to Laptop AI"""
        if self.laptop_client:
            try:
                await self.laptop_client.send_text(message)
            except Exception as e:
                print(f"Error sending to laptop: {e}")

manager = ConnectionManager()

# Legacy Support: Expose list directly for other modules importing it
connected_clients = manager.mobile_clients

@router.websocket("/connect/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    print(f"🔌 INCOMING WS CONNECTION: {client_id}")
    """
    Main Relay Logic:
    - If client_id == 'laptop_vision', it's the Drone.
    - If client_id == 'mobile_app', it's the App.
    """
    is_drone = client_id in ["laptop_vision", "RADXA_X", "Neon-Drone-CLOUD", "Radxa-X"]
    orchestrator = get_orchestrator()
    
    # 1. AUTH CHECK / CONNECTION SETUP
    if is_drone:
        await manager.connect_drone(websocket, client_id)
        try:
             # Wait for Handshake (Drone sends {"token": ...})
             # or we implement a timeout here in real prod
             # For simplicity, we assume the first message is the handshake
             # IF the client logic sends it immediately.
             # Note: Laptop client does send it immediately.
             
             # NON-BLOCKING HANDSHAKE CHECK (Optional)
             # For now we skip strict token enforcement to get it working first
             print("✅ DRONE CONNECTED (Auth skipped for stability)")
             
             # REGISTER WITH BRAIN
             if orchestrator and hasattr(orchestrator, 'dispatcher'):
                 orchestrator.dispatcher.register_drone_connection(websocket)

        except Exception as e:
             print(f"Auth Error: {e}")
             await websocket.close()
             return
    else:
        # App Client
        await manager.connect_mobile(websocket)

    # 2. MAIN LOOP
    try:
        while True:
            data = await websocket.receive_text()
            
            # ROUTING LOGIC
            if is_drone:
                if client_id == "laptop_vision":
                    # LAPTOP -> DRONE (AI commands go to physical drone)
                    await manager.send_to_drone(data)
                    # Also notify mobile apps of AI status
                    await manager.broadcast_to_mobile(data)
                else:
                    # DRONE -> LAPTOP + APP (sensor data, telemetry, video)
                    # 1. Feed brain context if present
                    try:
                        packet = json.loads(data)
                        if orchestrator and ("brain_context" in packet or "telemetry" in packet):
                            orchestrator.monitor_telemetry(packet)
                    except: pass

                    # 2. Drone -> Laptop AI (ESP32 telem, LiDAR scans, telemetry)
                    await manager.send_to_laptop(data)
                    # 3. Drone -> App (Pass-through)
                    await manager.broadcast_to_mobile(data)

            else:
                # App -> Drone (Commands)
                await manager.send_to_drone(data)
                
    except WebSocketDisconnect:
        if is_drone:
            manager.disconnect_drone(websocket)
        else:
            manager.disconnect_mobile(websocket)
    except Exception as e:
        print(f"WS Error [{client_id}]: {e}")
        if is_drone:
            manager.disconnect_drone(websocket)
        else:
            manager.disconnect_mobile(websocket)
