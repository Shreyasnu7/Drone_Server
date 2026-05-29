# File: laptop_ai/autonomous_director.py
"""
Autonomous Director
===================

High-level autonomous decision making for the drone.
Wraps FollowBrain and other autonomous policies.
"""
from laptop_ai.follow_brain import FollowBrain
try:
    from laptop_ai.follow_policy import FollowPolicy
except ImportError:
    FollowPolicy = None

class AutonomousDirector:
    def __init__(self):
        print("🧠 Initializing Autonomous Director...")
        self.follow_brain = FollowBrain()
        self.policy = FollowPolicy() if FollowPolicy else None
        print("✅ Autonomous Director: FollowBrain & Policy Wired")

    def update(self, environment_state, target_metadata=None, gemini_decision=None, er_decision=None):
        """
        Decide on next autonomous action based on ALL available intelligence.

        Priority:
        1. Local ER Brain decision (frame-by-frame, fastest)
        2. Gemini Live Brain decision (strategic, every 2s)
        3. FollowBrain (visual servoing when target is tracked)
        4. Hover (safe default when nothing is available)

        The AI decides freely — no hardcoded action restrictions.
        """
        # 1. Local ER Brain has frame-by-frame decisions (20-30 FPS)
        if er_decision and 'flight' in er_decision:
            flight = er_decision['flight']
            if not flight.get('hover', False):
                return {
                    "action": "ER_PILOT",
                    "flight": flight,
                    "gimbal": er_decision.get('gimbal', {}),
                    "camera_ai": er_decision.get('camera_ai', {}),
                    "reasoning": er_decision.get('reasoning', 'Local ER autonomous piloting')
                }

        # 2. Gemini Live Brain has strategic direction (every 2s)
        if gemini_decision and 'er_intent' in gemini_decision:
            return {
                "action": "GEMINI_DIRECTOR",
                "intent": gemini_decision['er_intent'],
                "camera_settings": gemini_decision.get('basic_camera_settings', {}),
                "reasoning": gemini_decision.get('reasoning', 'Gemini Live strategic direction')
            }

        # 3. FollowBrain — visual servoing when we have a tracked target
        if self.follow_brain and target_metadata:
            try:
                follow_cmd = self.follow_brain.update(
                    subject_bbox=target_metadata.get('bbox'),
                    frame_shape=target_metadata.get('frame_shape', (1080, 1920)),
                    current_distance=target_metadata.get('distance', 5.0)
                )
                if follow_cmd:
                    return {
                        "action": "FOLLOW",
                        "flight": follow_cmd,
                        "reasoning": "FollowBrain visual servoing on tracked target"
                    }
            except Exception as e:
                pass  # Fall through to hover

        # 4. Safe default — hover when no intelligence available
        return {"action": "HOVER", "reasoning": "No active brain or target — holding position"}
