from .intent_builder import IntentBuilder
from .cinematic_reasoner import CinematicReasoner
from .plan_generator import PlanGenerator
from ai.shot_intent.reasoning.intent_reasoner import ShotIntentReasoner
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from ai_context_store import context_store
except ImportError:
    # Fallback: inline minimal context store if module not found
    class _FallbackStore:
        def __init__(self):
            self._ctx = {}
        def update(self, data): self._ctx = data
        def get_context(self): return self._ctx
    context_store = _FallbackStore()
    print("⚠️ ai_context_store not found, using inline fallback")
import json

class CloudOrchestrator:
    """
    The Master Conductor.
    Pipeline:
    1. Multi-modal Ingest (Text/Image/Video) -> ShotIntentReasoner
    2. Cinematic Refinement -> CinematicReasoner
    3. Plan Generation -> PlanGenerator
    """
    def __init__(self, llm_client=None):
        self.intent_reasoner = ShotIntentReasoner(llm_client, "prompts/shot_intent.txt") if llm_client else None
        self.cinematic_reasoner = CinematicReasoner()
        self.planner = PlanGenerator()

    async def process_request(self, payload: dict) -> dict:
        """Full Pipeline Execution"""
        user_text = payload.get("text", "")
        
        # TELEMETRY INJECTION (Context-Aware AI)
        telemetry = payload.get("telemetry", {})
        if telemetry:
            user_text += f"\n[SYSTEM CONTEXT: Altitude={telemetry.get('altitude', 0)}m, Battery={telemetry.get('battery', 0)}%, GPS=({telemetry.get('lat', 0)}, {telemetry.get('lng', 0)})]"
        
        # LAPTOP AI CONTEXT INJECTION (Vision-Aware AI)
        laptop_ctx = context_store.get_context()
        if laptop_ctx:
            user_text += f"\n[LAPTOP_AI_CONTEXT]: {json.dumps(laptop_ctx, default=str)}"
        
        references = payload.get("media", []) 
        
        # Extract Dynamic Keys
        api_keys = payload.get("api_keys", {})

        # 1. RAW REASONING (The "Director" Brain)
        if self.intent_reasoner:
            raw_intent = self.intent_reasoner.reason(
                user_text=user_text,
                image_refs=[r for r in references if r['type'] == 'image'],
                video_refs=[r for r in references if r['type'] == 'video'],
                api_keys=api_keys
            )
        else:
            # Fallback if no LLM client passed (or mock)
            raw_intent = {"emotional_model": {"vector": {"neutral": 1.0}, "peak_allowed": True}, "camera_plan": {"shot_energy": 0.5}}

        # 2. CINEMATIC REFINEMENT (The "Cinematographer" Brain)
        # Adjusts energy, pacing, ensures we don't peak too early
        refined_intent = self.cinematic_reasoner.refine(raw_intent)

        # 3. PLAN GENERATION (The "Assistant Director" Brain)
        # Converts abstract intent into specific Drone Waypoints/Commands
        final_plan = self.planner.generate(refined_intent)

        return final_plan
