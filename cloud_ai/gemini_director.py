import os
import time
import json
import requests

class GeminiDirector:
    """
    Gemini 3.0 Cloud Logic Director.
    Handles high-level reasoning, intent classification, and complex mission planning.
    """
    def __init__(self, api_key=None):
        self.api_key = os.getenv("GEMINI_API_KEY") # SCRUBBED: Use Env Var
        self.model = "gemini-2.0-flash" # High rate limit model
        if not self.api_key:
             print("⚠️ No GEMINI_API_KEY in Environment. AI features will be limited.")
        print(f"✅ Gemini Director Initialized (Model: {self.model})")

    def ask_strategy(self, user_prompt, scene_context):
        """
        Queries Gemini for flight strategy based on scene description.
        REAL IMPLEMENTATION via REST API.
        """
        if not self.api_key:
             print("⚠️ No Gemini API Key. Using Fallback Rules.")
             return self._fallback_logic(user_prompt)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # Construct Payload
        scene_str = json.dumps(scene_context)
        system_prompt = "You are an autonomous drone director, you have to analyze the enviroment using the live sensor and video feed and think like a true movie crew would think, analyze the enviroment and lighing camera angles drone positions, path of drone and all the important aspects to correctly move the drone and gimbal, and do smart obstacle avoidance using the live camera and sensor feed. Output ONLY JSON."
        full_prompt = f"{system_prompt}\nUser: {user_prompt}\nScene: {scene_str}\nOutput JSON {{mode, params, cinematic_style: {{lut_name: str, contrast: float, saturation: float}}}}."
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }]
        }
        
        print(f"🧠 Gemini Thinking (Cloud Request)...")
        try:
            response = requests.post(url, json=payload, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                # Extract JSON
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != -1:
                    print("✅ Gemini Strategy Received.")
                    return json.loads(text[start:end])
        except Exception as e:
            print(f"❌ Gemini Cloud Error: {e}")
            
        return self._fallback_logic(user_prompt)

    def _fallback_logic(self, text):
        # Basic parsing if Cloud fails
        text = text.lower()
        if "orbit" in text: return {"mode": "orbit", "params": {"radius": 5.0}}
        if "follow" in text: return {"mode": "follow", "params": {"distance": 3.0}}
        return {"mode": "hover", "params": {}}
