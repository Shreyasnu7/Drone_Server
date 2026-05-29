from fastapi import APIRouter, HTTPException
from cloud_ai.orchestrator import CloudOrchestrator
from cloud_ai.llm import RealLLMClient
from plan_router import submit_plan
from api_schemas import DronePlan
import logging


# ADDED LOGGING TO DEBUG ROUTING
print("--> AI COMMAND ROUTER LOADED")

router = APIRouter(prefix="/director", tags=["AI-Command"])

# Initialize Orchestrator with REAL AI Client
# This ensures no fake "Mock" responses are ever generated.
orchestrator = CloudOrchestrator(llm_client=RealLLMClient())

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
        # AI is FREE to decide ANY action. We send the full plan to the drone
        # as-is. The drone's bridge handles MAVLink conversion.
        # No filtering, no hardcoded action lists — AI has full authority.
        from ws_router import manager
        import json

        # Build the dispatch message from the AI's plan
        # The action field is whatever the AI decided: ORBIT, FOLLOW, VELOCITY,
        # MOVE, GOTO, DOLLY, CRANE, CIRCLE, YAW, GIMBAL, TAKEOFF, LAND, etc.
        dispatch_msg = {
            "type": "ai_plan",
            "payload": {
                "action": plan.action,
                "params": plan.params or {},
                "confidence": plan.confidence,
                "reasoning": plan.reasoning,
                "emotional_context": plan.emotional_context or {},
            }
        }

        # Send to drone (Radxa bridge) — this is the actual execution target
        if manager.drone_client:
            print(f">>> AI DISPATCH TO DRONE: {plan.action} | params={plan.params}")
            try:
                await manager.drone_client.send_text(json.dumps(dispatch_msg))
            except Exception as e:
                print(f"Drone dispatch failed: {e}")

        # Also notify mobile apps so they see what the AI decided
        try:
            notify_msg = json.dumps({
                "type": "ai_status",
                "payload": {
                    "action": plan.action,
                    "reasoning": plan.reasoning,
                    "confidence": plan.confidence,
                }
            })
            await manager.broadcast_to_mobile(notify_msg)
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
