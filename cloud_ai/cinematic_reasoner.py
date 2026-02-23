# server/cloud_ai/cinematic_reasoner.py

class CinematicReasoner:
    """
    Applies cinematic intelligence:
    - pacing
    - emotional shaping
    - peak suppression
    """

    def refine(self, intent: dict) -> dict:
        # validate structure
        if "emotional_model" not in intent:
             intent["emotional_model"] = {"vector": {"neutral": 1.0}, "peak_allowed": True}
        if "camera_plan" not in intent:
             intent["camera_plan"] = {"shot_energy": 0.5}
        
        emotion = intent["emotional_model"].get("vector", {"neutral": 1.0})
        
        # Force all emotion values to float (Gemini sometimes returns strings)
        if isinstance(emotion, dict):
            emotion = {k: float(v) if not isinstance(v, (int, float)) else v for k, v in emotion.items()}
            intent["emotional_model"]["vector"] = emotion

        # Force shot_energy to float
        if "shot_energy" in intent.get("camera_plan", {}):
            try:
                intent["camera_plan"]["shot_energy"] = float(intent["camera_plan"]["shot_energy"])
            except (ValueError, TypeError):
                intent["camera_plan"]["shot_energy"] = 0.5

        # Prevent early climax
        if not intent["emotional_model"].get("peak_allowed", True):
            emotion = {
                k: min(float(v), 0.85) for k, v in emotion.items()
            }
        
        # Ensure shot_energy exists
        if "shot_energy" not in intent["camera_plan"]:
            intent["camera_plan"]["shot_energy"] = 0.5

        # Shape shot energy based on emotion
        if emotion:
            dominant = max(emotion, key=lambda k: float(emotion[k]))

            if dominant in ("awe", "grief"):
                intent["camera_plan"]["shot_energy"] *= 0.85
            elif dominant in ("tension", "urgency"):
                intent["camera_plan"]["shot_energy"] *= 1.1

        # Clamp values
        intent["camera_plan"]["shot_energy"] = max(
            0.1, min(float(intent["camera_plan"]["shot_energy"]), 1.0)
        )

        return intent