"""
ai/camera_brain/laptop_ai/ai_depth_estimator.py — Monocular depth estimation.

Uses MiDaS (Intel ISL) for real-time depth map generation from a single camera frame.
Falls back to a simple gradient if MiDaS/torch is not available.

Models (auto-selected by available compute):
- MiDaS v3.1 DPT-Large: Best quality, ~200ms/frame on RTX GPU
- MiDaS v3.1 DPT-Hybrid: Good balance, ~100ms/frame on RTX GPU
- MiDaS v2.1 Small: Fastest, ~30ms/frame, works on CPU

Depth output is RELATIVE (not metric) — normalized 0..1 where:
    0.0 = farthest (sky/background)
    1.0 = closest (foreground)

Used by:
- ai_autofocus.py: Focus distance estimation
- ai_camera_brain.py: Depth-aware exposure/composition
- obstacle_warp.py: Monocular obstacle detection (when no ToF/LiDAR)
"""
import numpy as np
import logging
import time

logger = logging.getLogger('depth_estimator')

# Try to import torch + MiDaS
_MIDAS_AVAILABLE = False
_torch = None
_midas_model = None
_midas_transform = None
_midas_device = None

try:
    import torch
    _torch = torch
    _MIDAS_AVAILABLE = True
    logger.info(f"PyTorch available: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    logger.warning("PyTorch not available — using fallback depth estimation")


class AIDepthEstimator:
    """
    Monocular depth estimator using MiDaS.

    Usage:
        estimator = AIDepthEstimator()
        depth_map, subject_mask = estimator.estimate(frame)
        # depth_map: float32 array [H, W], normalized 0..1
        # subject_mask: uint8 array [H, W], 255 = likely foreground subject
    """

    # Model selection (in order of preference)
    MODEL_CONFIGS = {
        'DPT_Large':  {'type': 'DPT_Large',  'transform': 'dpt_transform'},
        'DPT_Hybrid': {'type': 'DPT_Hybrid', 'transform': 'dpt_transform'},
        'MiDaS_small': {'type': 'MiDaS_small', 'transform': 'small_transform'},
    }

    def __init__(self, model_type=None, use_gpu=True):
        """
        Args:
            model_type: 'DPT_Large', 'DPT_Hybrid', or 'MiDaS_small'.
                        None = auto-select based on hardware.
            use_gpu: Whether to use CUDA if available.
        """
        self.model = None
        self.transform = None
        self.device = None
        self.model_type = model_type
        self.use_gpu = use_gpu
        self._initialized = False
        self._frame_count = 0
        self._total_time = 0.0

        if _MIDAS_AVAILABLE:
            self._init_midas()

    def _init_midas(self):
        """Load MiDaS model. Auto-selects best model for available hardware."""
        global _midas_model, _midas_transform, _midas_device

        try:
            # Select device
            if self.use_gpu and _torch.cuda.is_available():
                self.device = _torch.device('cuda')
                logger.info("Depth: Using CUDA GPU")
            else:
                self.device = _torch.device('cpu')
                logger.info("Depth: Using CPU")

            # Auto-select model
            if self.model_type is None:
                if _torch.cuda.is_available():
                    self.model_type = 'DPT_Large'
                else:
                    self.model_type = 'MiDaS_small'  # CPU-friendly

            logger.info(f"Loading MiDaS model: {self.model_type}")

            # Load via torch hub
            self.model = _torch.hub.load(
                'intel-isl/MiDaS', self.model_type,
                trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()

            # Load transforms
            midas_transforms = _torch.hub.load(
                'intel-isl/MiDaS', 'transforms',
                trust_repo=True
            )

            if self.model_type in ('DPT_Large', 'DPT_Hybrid'):
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self._initialized = True
            logger.info(f"MiDaS {self.model_type} loaded successfully")

        except Exception as e:
            logger.error(f"MiDaS init failed: {e} — falling back to gradient depth")
            self._initialized = False

    def estimate(self, frame):
        """
        Estimate depth from a BGR frame (OpenCV format).

        Args:
            frame: numpy array [H, W, 3] uint8 BGR

        Returns:
            depth: numpy array [H, W] float32, normalized 0..1
                   (1.0 = closest, 0.0 = farthest)
            mask: numpy array [H, W] uint8
                  (255 = likely foreground subject based on depth)
        """
        if frame is None:
            return np.zeros((480, 640), dtype=np.float32), np.zeros((480, 640), dtype=np.uint8)

        h, w = frame.shape[:2]

        if self._initialized:
            return self._estimate_midas(frame, h, w)
        else:
            return self._estimate_fallback(frame, h, w)

    def _estimate_midas(self, frame, h, w):
        """Real MiDaS depth estimation."""
        start = time.time()

        try:
            import cv2
            # Convert BGR → RGB for MiDaS
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply MiDaS transform
            input_batch = self.transform(rgb).to(self.device)

            # Inference
            with _torch.no_grad():
                prediction = self.model(input_batch)

                # Resize to original resolution
                prediction = _torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth_raw = prediction.cpu().numpy()

            # Normalize to 0..1 (MiDaS outputs inverse depth — higher = closer)
            depth_min = depth_raw.min()
            depth_max = depth_raw.max()
            if depth_max - depth_min > 0:
                depth = (depth_raw - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth_raw)

            # Generate subject mask (foreground = depth > 70th percentile)
            threshold = np.percentile(depth, 70)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[depth > threshold] = 255

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            elapsed = time.time() - start
            self._frame_count += 1
            self._total_time += elapsed

            if self._frame_count % 100 == 0:
                avg = self._total_time / self._frame_count
                logger.info(f"Depth: avg {avg*1000:.0f}ms/frame ({1/avg:.1f} FPS)")

            return depth.astype(np.float32), mask

        except Exception as e:
            logger.error(f"MiDaS inference failed: {e}")
            return self._estimate_fallback(frame, h, w)

    def _estimate_fallback(self, frame, h, w):
        """
        Fallback depth estimation without ML.
        Uses image cues: blur gradient, brightness, saturation.
        Better than a simple gradient but not as good as MiDaS.
        """
        try:
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # 1. Blur-based depth cue (blurrier = farther)
            sharp = cv2.Laplacian(gray, cv2.CV_32F)
            sharpness = np.abs(sharp)
            # Smooth the sharpness map
            sharpness = cv2.GaussianBlur(sharpness, (31, 31), 0)
            # Normalize
            s_min, s_max = sharpness.min(), sharpness.max()
            if s_max - s_min > 0:
                sharpness_norm = (sharpness - s_min) / (s_max - s_min)
            else:
                sharpness_norm = np.zeros_like(sharpness)

            # 2. Vertical position cue (lower = closer for aerial view)
            y_gradient = np.linspace(0.3, 1.0, h).reshape(h, 1)
            y_gradient = np.tile(y_gradient, (1, w))

            # 3. Center bias (subjects tend to be centered)
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            center_dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
            max_dist = np.sqrt(cx**2 + cy**2)
            center_bias = 1.0 - (center_dist / max_dist)

            # Combine cues (weighted)
            depth = (0.4 * sharpness_norm +
                     0.3 * y_gradient.astype(np.float32) +
                     0.3 * center_bias)

            # Normalize final depth
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 0:
                depth = (depth - d_min) / (d_max - d_min)

            # Subject mask from center + sharpness
            mask = np.zeros((h, w), dtype=np.uint8)
            threshold = np.percentile(depth, 70)
            mask[depth > threshold] = 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            return depth.astype(np.float32), mask

        except Exception:
            # Absolute last resort
            depth = np.tile(np.linspace(1, 0, w), (h, 1)).astype(np.float32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = w // 2, h // 2
            r = min(80, h // 4, w // 4)
            mask[max(0, cy-r):cy+r, max(0, cx-r):cx+r] = 255
            return depth, mask

    def get_depth_at_point(self, depth_map, x, y):
        """Get depth value at a specific pixel coordinate."""
        h, w = depth_map.shape[:2]
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        return float(depth_map[y, x])

    def get_depth_stats(self):
        """Performance statistics."""
        if self._frame_count == 0:
            return {'model': self.model_type, 'initialized': self._initialized, 'frames': 0}
        return {
            'model': self.model_type,
            'initialized': self._initialized,
            'frames': self._frame_count,
            'avg_ms': round(self._total_time / self._frame_count * 1000, 1),
            'device': str(self.device) if self.device else 'none',
        }