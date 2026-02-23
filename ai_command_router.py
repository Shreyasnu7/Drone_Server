from fastapi import APIRouter, HTTPException
from cloud_ai.orchestrator import CloudOrchestrator
from cloud_ai.llm import RealLLMClient
from plan_router import submit_plan
from api_schemas import DronePlan
from ai_context_store import context_store
import logging


# ADDED LOGGING TO DEBUG ROUTING
print("--> AI COMMAND ROUTER LOADED")

router = APIRouter(prefix="/director", tags=["AI-Command"])

# Initialize Orchestrator with REAL AI Client
orchestrator = CloudOrchestrator(llm_client=RealLLMClient())


@router.post("/ai/context")
async def receive_laptop_context(payload: dict):
    """
    Receives live vision context from Laptop AI.
    Called periodically by director_core.py on the laptop.
    """
    context_store.update(payload)
    return {"status": "ok", "age": 0}

@router.post("/ai/command")
async def ai_command(payload: dict):
    """
    Structured Pipeline: 
    Input -> Orchestrator -> Plan -> Queue
    """
    print(f"--> AI COMMAND HANDLER ENTERED. Payload Keys: {list(payload.keys())}")
    try:
        # 0. SYSTEM COMMAND BYPASS (For Config/Direct Control)
        if payload.get("provider") == "system":
            print(f"⚙️ SYSTEM COMMAND: {payload.get('text')}")
            # Direct Plan Creation (Skip LLM)
            # We wrap the text (e.g., "SET_CONFIG: ...") into a Plan Action
            # The Brain's director_core.py knows how to parse this string.
            plan = DronePlan(
                thought_process="System Override",
                reasoning="Direct User Configuration",
                action=payload.get("text"), # "SET_CONFIG: source=external"
                emotional_context={"override": 1.0},
                confidence=1.0
            )
            await submit_plan(plan)
            return {"status": "queued", "plan": plan.dict(), "type": "system"}

        # 1. Delegate to Orchestrator (Real AI)
        plan_result = await orchestrator.process_request(payload)
        
        # 2. Ensure it matches DronePlan schema
        if isinstance(plan_result, dict):
            plan = DronePlan(**plan_result)
        else:
            plan = plan_result

        # 3. Push to Execution Queue (Legacy/Laptop Polling)
        await submit_plan(plan)

        # 4. DIRECT DISPATCH (Cloud Mode)
        # Immediately send to connected drones via WebSocket
        from ws_router import connected_clients, WebSocket
        import json
        
        # Convert Plan -> MAVLink/Radxa Command
        # Plan Action: "GIMBAL_TRACK target=car" -> {"type": "gimbal", "payload": ...}
        # Plan Action: "TAKEOFF" -> {"type": "ai", "payload": {"action": "TAKEOFF"}}
        
        cmd_payload = {}
        cmd_type = "ai" # default
        
        if "GIMBAL" in plan.action.upper():
            cmd_type = "gimbal"
            # Extract Pitch/Yaw (Simplified - In real system, reasoning would give numbers)
            # Fore now, let's assume raw_intent usually has numerical data, but here we just pass intent
            # Actually, let's just send the whole plan and let Radxa decide? 
            # Radxa only understands specific JSON.
            
            # Parsing "GIMBAL_LOOK: 45, 0"
            try:
                parts = plan.action.split(":")
                if len(parts) > 1:
                    coords = parts[1].split(",")
                    cmd_payload = {"pitch": int(coords[0]), "yaw": int(coords[1])}
            except:
                cmd_payload = {"pitch": 0, "yaw": 0}

        elif "TAKEOFF" in plan.action.upper():
             cmd_type = "ai"
             cmd_payload = {"action": "TAKEOFF"}
        elif "LAND" in plan.action.upper():
             cmd_type = "ai"
             cmd_payload = {"action": "LAND"}
        
        # Broadcast to Drones
        if cmd_type and connected_clients:
            print(f"📡 DISPATCHING TO DRONES: {cmd_type} {cmd_payload}")
            msg = json.dumps({"type": cmd_type, "payload": cmd_payload})
            
            # connected_clients is a List[WebSocket], not a Dict
            for sock in connected_clients:
                 try:
                     await sock.send_text(msg)
                 except:
                     pass

        return {
            "status": "queued_and_dispatched",
            "plan": plan.dict()
        }
    except Exception as e:
        print(f"❌ ORCHESTRATOR ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ShotIntentMemory:
    @staticmethod
    def record_outcome(plan_id: str, success: bool, user_rating: int, comments: str):
        """
        Closes the loop. 
        In strict 'Learning' mode, this data would be fed back into the Fine-tuning dataset.
        For now, we store it in 'memory/learning_log.jsonl'.
        """
        import json
        import os
        from utils import BASE_DATA_DIR
        
        log_path = os.path.join(BASE_DATA_DIR, "memory", "learning_log.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        entry = {
            "timestamp": 1234567890, # TODO: Real time
            "plan_id": plan_id,
            "success": success,
            "rating": user_rating,
            "comments": comments
        }
        
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
@router.post("/feedback")
async def record_feedback(payload: dict):
    # payload: { "plan_id": "...", "success": true, "rating": 5, "comments": "Good job" }
    try:
        ShotIntentMemory.record_outcome(
            payload.get("plan_id", "unknown"),
            payload.get("success", True),
            payload.get("rating", 0),
            payload.get("comments", "")
        )
        return {"status": "recorded", "message": "AI Learning Loop Updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
