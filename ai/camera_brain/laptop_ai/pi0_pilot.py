"""
π0-FAST: Real Autonomous Pilot Module
50Hz flight control policy running on NVIDIA RTX 5070 Ti.

Model loading priority:
1. TensorRT engine (fastest, ~2ms inference)
2. ONNX via onnxruntime-gpu
3. PyTorch model (LeRobot/pi0-fast)
4. Fallback: PD controller (no AI, just proportional-derivative)
"""

import time
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Try TensorRT first (NVIDIA optimized)
trt_available = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    trt_available = True
except ImportError:
    pass

# Try ONNX Runtime (cross-platform GPU)
ort_available = False
try:
    import onnxruntime as ort
    ort_available = True
except ImportError:
    pass

# Try PyTorch (general)
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    pass


class Pi0Pilot:
    """
    π0-FAST: Specialized 50Hz Autonomous Pilot.
    
    Inputs (state vector):
        - target_err_x: Normalized target error X (-1 to 1)
        - target_err_y: Normalized target error Y (-1 to 1)
        - vx, vy: Current velocity (m/s)
        - depth_dist: Front obstacle distance (m)
        - altitude: Current altitude (m)
        - heading: Current heading (deg)
    
    Outputs:
        - roll: Roll command (degrees, -30 to 30)
        - pitch: Pitch command (degrees, -30 to 30)
        - yaw_rate: Yaw rate (deg/s)
        - throttle: Throttle (0.0 to 1.0)
    """

    # Base directory relative to this file
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Model file paths (searched in order)
    MODEL_PATHS = [
        os.path.join(_BASE_DIR, "models", "pi0_fast.engine"),        # TensorRT engine
        os.path.join(_BASE_DIR, "models", "pi0_fast.onnx"),          # ONNX model
        os.path.join(_BASE_DIR, "models", "pi0_fast.pt"),            # PyTorch checkpoint
    ]

    def __init__(self):
        self.control_rate = 50.0  # Hz
        self.last_update = time.time()
        self.last_state = {}
        self.model = None
        self.model_type = None  # "tensorrt", "onnx", "pytorch", "pd_fallback"
        
        # PD Controller gains (used as fallback AND for safety clamping)
        self.kp = 0.5
        self.ki = 0.01
        self.kd = 0.15
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        
        # Safety limits
        self.max_roll = 30.0    # degrees
        self.max_pitch = 30.0   # degrees
        self.max_yaw_rate = 90.0  # deg/s
        self.min_obstacle_dist = 1.5  # meters — emergency brake
        
        # Try to load real model
        self._load_model()
        
        mode_str = self.model_type.upper() if self.model_type else "NONE"
        print(f"✅ Pi0-FAST Pilot Loaded ({mode_str} mode, {self.control_rate}Hz)")

    def _load_model(self):
        """Try to load the best available model runtime."""
        
        for path in self.MODEL_PATHS:
            if not os.path.exists(path):
                continue
            
            # TensorRT Engine
            if path.endswith(".engine") and trt_available:
                try:
                    self.model = self._load_tensorrt(path)
                    self.model_type = "tensorrt"
                    logger.info(f"🚀 Pi0-FAST: TensorRT engine loaded from {path}")
                    return
                except Exception as e:
                    logger.warning(f"TensorRT load failed: {e}")
            
            # ONNX Runtime
            elif path.endswith(".onnx") and ort_available:
                try:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.model = ort.InferenceSession(path, providers=providers)
                    self.model_type = "onnx"
                    logger.info(f"🚀 Pi0-FAST: ONNX model loaded from {path} (GPU: {'CUDA' in str(self.model.get_providers())})")
                    return
                except Exception as e:
                    logger.warning(f"ONNX load failed: {e}")
            
            # PyTorch
            elif path.endswith(".pt") and torch_available:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    try:
                        self.model = torch.jit.load(path, map_location=device)
                    except:
                        self.model = torch.load(path, map_location=device)
                    self.model.eval()
                    self.model_type = "pytorch"
                    logger.info(f"🚀 Pi0-FAST: PyTorch model loaded from {path} (device: {device})")
                    return
                except Exception as e:
                    logger.warning(f"PyTorch load failed: {e}")
        
        # No model found — use PD fallback
        self.model_type = "pd_fallback"
        logger.warning("⚠️ Pi0-FAST: No neural model found. Using PID controller fallback.")
        logger.warning(f"   To enable neural pilot, place model at: {self.MODEL_PATHS}")

    def _load_tensorrt(self, engine_path):
        """Load TensorRT engine for fastest inference."""
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return {"engine": engine, "context": context}

    def update(self, state_vector: dict) -> dict:
        """
        Run the flight control policy at 50Hz.
        Returns: {roll, pitch, yaw_rate, throttle}
        """
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        # Extract state
        err_x = state_vector.get('target_err_x', 0.0)
        err_y = state_vector.get('target_err_y', 0.0)
        vx = state_vector.get('vx', 0.0)
        vy = state_vector.get('vy', 0.0)
        depth = state_vector.get('depth_dist', 999.0)
        altitude = state_vector.get('altitude', 0.0)
        heading = state_vector.get('heading', 0.0)
        
        # === SAFETY: Emergency obstacle braking ===
        if depth < self.min_obstacle_dist:
            logger.warning(f"🛑 Pi0 EMERGENCY: Obstacle at {depth:.1f}m — full brake")
            return {
                "roll": 0.0,
                "pitch": 5.0,  # Slight pull-back
                "yaw_rate": 0.0,
                "throttle": 0.5,  # Maintain altitude
                "emergency": True
            }
        
        # === RUN POLICY ===
        if self.model_type == "tensorrt":
            commands = self._infer_tensorrt(err_x, err_y, vx, vy, depth, altitude, heading)
        elif self.model_type == "onnx":
            commands = self._infer_onnx(err_x, err_y, vx, vy, depth, altitude, heading)
        elif self.model_type == "pytorch":
            commands = self._infer_pytorch(err_x, err_y, vx, vy, depth, altitude, heading)
        else:
            commands = self._pd_control(err_x, err_y, vx, vy, dt)
        
        # === SAFETY: Clamp outputs ===
        commands["roll"] = float(np.clip(commands["roll"], -self.max_roll, self.max_roll))
        commands["pitch"] = float(np.clip(commands["pitch"], -self.max_pitch, self.max_pitch))
        commands["yaw_rate"] = float(np.clip(commands.get("yaw_rate", 0), -self.max_yaw_rate, self.max_yaw_rate))
        commands["throttle"] = float(np.clip(commands.get("throttle", 0.5), 0.0, 1.0))
        commands["emergency"] = False
        
        self.last_state = state_vector
        return commands

    def _pd_control(self, err_x, err_y, vx, vy, dt):
        """PID controller fallback — works without any model."""
        dt = max(dt, 0.001)
        
        # Integral (with anti-windup)
        self.integral_x = np.clip(self.integral_x + err_x * dt, -5.0, 5.0)
        self.integral_y = np.clip(self.integral_y + err_y * dt, -5.0, 5.0)
        
        # Derivative
        d_err_x = (err_x - self.prev_err_x) / dt
        d_err_y = (err_y - self.prev_err_y) / dt
        self.prev_err_x = err_x
        self.prev_err_y = err_y
        
        roll = err_x * self.kp + self.integral_x * self.ki + d_err_x * self.kd
        pitch = err_y * self.kp + self.integral_y * self.ki + d_err_y * self.kd
        
        return {
            "roll": roll * 30.0,   # Scale to degrees
            "pitch": pitch * 30.0,
            "yaw_rate": 0.0,
            "throttle": 0.5
        }

    def _infer_tensorrt(self, err_x, err_y, vx, vy, depth, alt, heading):
        """Run TensorRT engine inference."""
        ctx = self.model["context"]
        engine = self.model["engine"]
        
        # Prepare input tensor
        input_data = np.array([[err_x, err_y, vx, vy, depth, alt, heading]], dtype=np.float32)
        
        # Allocate device memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        output_data = np.empty([1, 4], dtype=np.float32)  # [roll, pitch, yaw, throttle]
        d_output = cuda.mem_alloc(output_data.nbytes)
        
        # Transfer input, run, transfer output
        cuda.memcpy_htod(d_input, input_data)
        ctx.execute_v2(bindings=[int(d_input), int(d_output)])
        cuda.memcpy_dtoh(output_data, d_output)
        
        return {
            "roll": float(output_data[0][0]),
            "pitch": float(output_data[0][1]),
            "yaw_rate": float(output_data[0][2]),
            "throttle": float(output_data[0][3])
        }

    def _infer_onnx(self, err_x, err_y, vx, vy, depth, alt, heading):
        """Run ONNX Runtime inference."""
        input_data = np.array([[err_x, err_y, vx, vy, depth, alt, heading]], dtype=np.float32)
        input_name = self.model.get_inputs()[0].name
        result = self.model.run(None, {input_name: input_data})
        out = result[0][0]
        return {
            "roll": float(out[0]),
            "pitch": float(out[1]),
            "yaw_rate": float(out[2]),
            "throttle": float(out[3])
        }

    def _infer_pytorch(self, err_x, err_y, vx, vy, depth, alt, heading):
        """Run PyTorch model inference."""
        device = next(self.model.parameters()).device
        x = torch.tensor([[err_x, err_y, vx, vy, depth, alt, heading]], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = self.model(x).cpu().numpy()[0]
        return {
            "roll": float(out[0]),
            "pitch": float(out[1]),
            "yaw_rate": float(out[2]),
            "throttle": float(out[3])
        }
