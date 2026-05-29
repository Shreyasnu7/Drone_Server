from fastapi import APIRouter, Response, File, UploadFile, Request
from fastapi.responses import StreamingResponse
import asyncio
import time
from typing import List, Dict

router = APIRouter()

# Global Frame Buffer (In-Memory, Single Stream)
latest_frame = None

@router.post("/video_push")
async def video_push(file: bytes = File(...)):
    """Radxa pushes frames here"""
    global latest_frame
    latest_frame = file
    return {"status": "ok"}

def get_video_stream():
    """Generator for MJPEG stream"""
    global latest_frame
    while True:
        if latest_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        else:
             # Yield specific placeholder or wait
             pass
        time.sleep(0.04) # ~25 FPS limit

@router.get("/video_feed")
async def video_feed():
    """App consumes stream here"""
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/drones")
async def get_drones() -> List[Dict]:
    """
    Returns list of active drones based on Real-Time WS Connections.
    STRICTLY REAL: No hardcoded entries.
    """
    drones = []
    
    # Real Scan
    try:
        from ws_router import manager
        if manager.drone_client:
             drones.append({
                "id": "drone_1",
                "name": "Neon-Drone-X (Online)",
                "status": "Online",
                "connection": "WS-Active"
             })
    except ImportError:
        pass # Fallback or Main not reachable

    return drones

@router.get("/media")
async def get_media(request: Request):
    """
    Returns list of media files (photos/videos) from the server's storage.
    Real implementation scans a 'media' directory.
    """
    import os
    media_dir = "media"
    if not os.path.exists(media_dir):
        os.makedirs(media_dir, exist_ok=True)
        
    files = []
    base_url = str(request.base_url).rstrip('/')
    
    for f in os.listdir(media_dir):
        if f.endswith(('.jpg', '.png', '.mp4', '.avi')):
             url = f"{base_url}/media/{f}" 
             ftype = "video" if f.endswith(('.mp4', '.avi')) else "photo"
             # Real modification time
             try:
                 stats = os.stat(os.path.join(media_dir, f))
                 date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime))
             except: date_str = "Unknown"
             
             files.append({"url": url, "type": ftype, "date": date_str})
             
    return files

@router.post("/media/upload")
async def upload_media(file: UploadFile = File(...)):
    """
    Accepts photo/video uploads from Radxa/Drone.
    Saves to 'media/' directory.
    """
    import os
    import shutil
    
    media_dir = "media"
    if not os.path.exists(media_dir):
        os.makedirs(media_dir, exist_ok=True)
        
    file_location = os.path.join(media_dir, file.filename)
    try:
        with open(file_location, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Media Uploaded: {file.filename}")
        return {"status": "ok", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/command")
async def generic_command(payload: Dict):
    """
    Relays generic APP commands (set_rth, etc.) to connected drones.
    Input: {"command": "set_rth_alt", "payload": {"alt": 5000}}
    """
    from ws_router import manager
    import json
    
    cmd = payload.get("command")
    data = payload.get("payload")
    
    # MA-33 FIX: Use type="command" so bridge command_loop recognizes it
    # (was "ai" which bridge doesn't handle)
    msg = json.dumps({"type": "command", "payload": cmd, **({"params": data} if data else {})})
    
    try:
        await manager.send_to_drone(msg)
        return {"status": "dispatched", "drones_reached": 1}
    except Exception as e:
        return {"status": "error", "message": str(e)}
