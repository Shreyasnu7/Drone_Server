# File: laptop_ai/ai_color_engine.py
"""
AI Color Engine
- Suggests color grading parameters and applies simple auto-grade transforms.
- Supports:
    - auto white balance (AWB) estimate
    - auto contrast/gain
    - film LUT placeholder (returns identifier)
    - quick LOG -> Rec.709 linearization stub
- DOES NOT ship heavy LUTs or proprietary transforms; provides interfaces you can extend.

Public class: AIColorEngine
Methods:
    - analyze(frame) -> stats
    - propose_grade(stats) -> grade dict
    - apply_grade(frame, grade) -> graded_frame
Dependencies: numpy, opencv
"""
from __future__ import annotations
import time
from typing import Dict, Tuple
import numpy as np
import cv2
import cv2
# RE-WIRED (Audit Fix)
from laptop_ai.ai_colourist import AIColorist

def simple_awb_grayworld(bgr):
    # Gray-world white balance correction
    b,g,r = cv2.split(bgr.astype(np.float32))
    mb = b.mean()
    mg = g.mean()
    mr = r.mean()
    # scale channels to match green
    kb = mg / (mb+1e-9)
    kr = mg / (mr+1e-9)
    b = b * kb
    r = r * kr
    out = cv2.merge([b,g,r])
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, (kb,1.0,kr)

class AIColorEngine:
    def __init__(self):
        # RE-WIRED (Audit Fix)
        try:
             self.colorist = AIColorist()
        except:
             self.colorist = None
             
        self.lut_catalog = ["cine_soft", "cine_flat", "vivid_boost"] 
        self.load_user_luts() # Scan for User's "1000 Files"
        
        # CRITICAL: Import the 200K+ line global_tone_curve masterpiece!
        self.tone_curve = None
        try:
            from laptop_ai.global_tone_curve import GlobalToneCurveEngine
            self.tone_curve = GlobalToneCurveEngine()
            print("🎨 LOADED: global_tone_curve.py (200K+ lines) - Crown Jewel Active!")
        except Exception as e:
            print(f"⚠️ global_tone_curve load error: {e}")
            self.tone_curve = None

    def load_user_luts(self):
        """Scans for user-provided .cube / .h5 files"""
        import os
        asset_dir = os.path.join(os.path.dirname(__file__), "ai_pipeline", "assets")
        if os.path.exists(asset_dir):
            for f in os.listdir(asset_dir):
                if f.endswith(('.cube', '.h5')):
                     self.lut_catalog.append(f)
            print(f"🎨 Loaded {len(self.lut_catalog)} Cinematic Styles (User + Built-in)")

    def analyze(self, frame: np.ndarray) -> Dict:
        """
        Return color stats (mean RGB, contrast, histogram peaks)
        """
        mean = frame.mean(axis=(0,1)).tolist()
        std = frame.std(axis=(0,1)).tolist()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_mean = float(hsv[:,:,2].mean())/255.0
        return {
            "mean_rgb": mean,
            "std_rgb": std,
            "v_mean": v_mean,
            "ts": time.time()
        }

    def propose_grade(self, stats: Dict, style: str = "cine_soft") -> Dict:
        """
        Propose a grade configuration: {"awb_scales": (kb,kg,kr), "lut": str, "contrast": float, "saturation": float}
        """
        # For low light increase saturation and contrast a bit, pick LUT accordingly
        v = stats.get("v_mean", 0.5)
        if v < 0.35:
            lut = "cine_flat"
            contrast = 0.9
            saturation = 1.15
        elif v > 0.7:
            lut = "vivid_boost"
            contrast = 1.05
            saturation = 1.05
        else:
            lut = style
            contrast = 1.0
            saturation = 1.0
        return {
            "lut": lut,
            "contrast": float(contrast),
            "saturation": float(saturation),
            "ts": time.time()
        }

        """
        Apply a light grade: AWB + contrast + saturation (fast)
        This is a lightweight, real-time-friendly grading.
        """
        # AWB (simple gray-world)
        wb_frame, scales = simple_awb_grayworld(frame)
        # convert to float
        img = wb_frame.astype(np.float32) / 255.0
        # adjust contrast and saturation
        contrast = grade.get("contrast", 1.0)
        saturation = grade.get("saturation", 1.0)
        # contrast: simple scale around 0.5
        img = (img - 0.5) * contrast + 0.5
        # saturation: convert to HSV and scale V and S
        hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out

# Example usage:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ce = AIColorEngine()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        stats = ce.analyze(frame)
        grade = ce.propose_grade(stats)
        out = ce.apply_grade(frame, grade)
        cv2.imshow("graded", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
