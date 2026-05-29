# server/cloud_ai/plan_generator.py

from api_schemas import DronePlan


class PlanGenerator:
    """
    Converts AI reasoning output into a DronePlan.
    """

    def generate(self, ai_result: dict) -> DronePlan:
        # ROBUST MAPPING: Try 'action' first, then 'intent_type'
        raw_action = ai_result.get("action") or ai_result.get("intent_type") or "hover"
        
        return DronePlan(
            action=raw_action,
            thought_process=ai_result.get("thought_process", "Processing..."),
            reasoning=ai_result.get("reasoning", "AI decision"),
            # MA-22 FIX: LLM may return params under different keys depending on reasoning
            params=ai_result.get("params") or ai_result.get("constraints") or ai_result.get("motion_plan") or ai_result.get("laptop_ai_directives") or {},
            emotional_context=ai_result.get("emotional_context", {}),
            confidence=ai_result.get("confidence", 1.0)
        )