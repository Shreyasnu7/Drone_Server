"""
raxda/ai/vision.py — Lightweight onboard computer vision for Radxa Zero 3W.

Runs on the drone's onboard SBC. NOT the heavy laptop AI — this is a
lightweight fallback that works when cloud/laptop connection is lost.

Capabilities:
- Object detection using YOLOv8 nano (or motion detection as fallback)
- Subject tracking (centroid-based, lightweight for Radxa's CPU)
- Obstacle detection from camera frame (large objects in lower frame)
- Landing zone safety check (clear ground detection)
- Basic scene classification (indoor/outdoor/crowded)

Hardware: Radxa Zero 3W (4GB RAM, Mali-G52 GPU)
Camera: IMX219 (Pi Camera V2) via GStreamer, or GoPro UDP stream

Data Flow:
    Camera frame ──▶ VisionProcessor.process() ──▶ {detections, tracking, scene}
                                                    ▶ fed to decision.py
"""
import time
import logging
import numpy as np

logger = logging.getLogger('raxda.vision')

# Try to import CV2 and YOLO — graceful fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available — vision disabled")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not available — using basic CV detection")


class Detection:
    """Single detected object."""
    __slots__ = ['class_id', 'class_name', 'confidence', 'bbox', 'center', 'area']

    def __init__(self, class_id, class_name, confidence, bbox):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


class SimpleTracker:
    """
    Lightweight centroid tracker — suitable for Radxa's limited compute.
    Tracks objects by matching centroids across frames using distance threshold.
    """

    def __init__(self, max_disappeared=10, max_distance=80):
        self.next_id = 0
        self.objects = {}       # track_id -> centroid (cx, cy)
        self.disappeared = {}   # track_id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        """
        Update tracker with new detections. Returns dict of track_id -> Detection.
        """
        if not detections:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]
            return {}

        new_centroids = [d.center for d in detections]

        if not self.objects:
            tracked = {}
            for i, det in enumerate(detections):
                self.objects[self.next_id] = det.center
                self.disappeared[self.next_id] = 0
                tracked[self.next_id] = det
                self.next_id += 1
            return tracked

        # Greedy matching by closest centroid (no Hungarian — too heavy for Radxa)
        existing_ids = list(self.objects.keys())
        existing_centroids = [self.objects[tid] for tid in existing_ids]
        used_detections = set()
        tracked = {}

        for i, tid in enumerate(existing_ids):
            ec = existing_centroids[i]
            best_dist = self.max_distance
            best_j = -1
            for j, nc in enumerate(new_centroids):
                if j in used_detections:
                    continue
                dist = ((ec[0] - nc[0])**2 + (ec[1] - nc[1])**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                self.objects[tid] = new_centroids[best_j]
                self.disappeared[tid] = 0
                tracked[tid] = detections[best_j]
                used_detections.add(best_j)
            else:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]

        # Register unmatched detections as new tracks
        for j, det in enumerate(detections):
            if j not in used_detections:
                self.objects[self.next_id] = det.center
                self.disappeared[self.next_id] = 0
                tracked[self.next_id] = det
                self.next_id += 1

        return tracked


class VisionProcessor:
    """
    Main vision processor for onboard use.

    Usage:
        vp = VisionProcessor()
        vp.start_camera()

        while running:
            result = vp.process()
            # result = {
            #     'detections': [...],
            #     'tracked': {id: Detection},
            #     'scene': 'outdoor',
            #     'landing_safe': True,
            #     'fps': 12.5,
            # }
    """

    RELEVANT_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        5: 'bus', 7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog',
    }

    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.4, camera_source=0):
        self.conf_threshold = conf_threshold
        self.camera_source = camera_source
        self.cap = None
        self.model = None
        self.tracker = SimpleTracker()
        self.frame_count = 0
        self.fps = 0.0
        self._fps_timer = time.time()
        self._last_scene = 'unknown'

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                logger.info(f"YOLOv8 loaded: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self.model = None

    def start_camera(self, source=None):
        """
        Start camera capture. Tries GStreamer (Pi Camera) first,
        then V4L2, then UDP (GoPro stream).
        """
        if not CV2_AVAILABLE:
            logger.error("Cannot start camera: OpenCV not installed")
            return False

        src = source or self.camera_source

        # Priority 1: GStreamer pipeline for IMX219
        gst_pipeline = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! appsink"
        )
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if self.cap.isOpened():
            logger.info("Camera started: GStreamer (IMX219)")
            return True

        # Priority 2: Direct V4L2
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.info("Camera started: V4L2")
            return True

        # Priority 3: UDP stream from GoPro
        self.cap = cv2.VideoCapture("udp://@0.0.0.0:8554")
        if self.cap.isOpened():
            logger.info("Camera started: UDP GoPro stream")
            return True

        logger.error("No camera source available")
        return False

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def grab_frame(self):
        """Grab a single frame. Returns numpy array or None."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def detect(self, frame):
        """Run object detection on a frame. Returns list of Detection."""
        if frame is None:
            return []

        detections = []

        if self.model is not None:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in self.RELEVANT_CLASSES:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    det = Detection(
                        class_id=cls_id,
                        class_name=self.RELEVANT_CLASSES[cls_id],
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                    detections.append(det)
        else:
            detections = self._fallback_detect(frame)

        return detections

    def _fallback_detect(self, frame):
        """
        Basic detection without ML — background subtraction + contour detection.
        Detects large moving objects (people, vehicles).
        """
        if not CV2_AVAILABLE:
            return []

        if not hasattr(self, '_bg_sub'):
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=40, detectShadows=False
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fg_mask = self._bg_sub.apply(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            det = Detection(
                class_id=-1,
                class_name='moving_object',
                confidence=min(area / 5000.0, 0.95),
                bbox=(x, y, x + w, y + h),
            )
            detections.append(det)

        return detections

    def classify_scene(self, frame):
        """
        Lightweight scene classification based on color histogram + edge density.
        Returns: 'outdoor_open', 'outdoor_urban', 'indoor', 'crowded', 'dark'
        """
        if frame is None or not CV2_AVAILABLE:
            return 'unknown'

        h, w = frame.shape[:2]
        roi = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            self._last_scene = 'dark'
            return 'dark'

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        upper = frame[:h // 3]
        hsv_upper = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_upper, (90, 30, 100), (130, 255, 255))
        sky_ratio = np.sum(blue_mask > 0) / blue_mask.size

        if sky_ratio > 0.3 and edge_density < 0.1:
            scene = 'outdoor_open'
        elif sky_ratio > 0.1 and edge_density > 0.15:
            scene = 'outdoor_urban'
        elif edge_density > 0.2:
            scene = 'indoor'
        else:
            scene = 'outdoor_open'

        self._last_scene = scene
        return scene

    def check_landing_zone(self, frame):
        """
        Check if the area below the drone is safe for landing.
        Returns: (is_safe: bool, reason: str)
        """
        if frame is None or not CV2_AVAILABLE:
            return False, 'no_frame'

        h, w = frame.shape[:2]
        ground = frame[2 * h // 3:, :]
        gray = cv2.cvtColor(ground, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        if edge_ratio > 0.25:
            return False, f'complex_terrain (edges={edge_ratio:.2f})'

        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.sum(thresh > 0) / thresh.size
        if dark_ratio > 0.3:
            return False, f'obstacles_detected (dark_ratio={dark_ratio:.2f})'

        return True, 'clear'

    def process(self):
        """
        Full pipeline: grab frame → detect → track → classify.
        Returns comprehensive result dict.
        """
        frame = self.grab_frame()

        self.frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self._fps_timer = time.time()

        if frame is None:
            return {
                'detections': [],
                'tracked': {},
                'scene': self._last_scene,
                'landing_safe': False,
                'landing_reason': 'no_frame',
                'fps': self.fps,
                'frame': None,
            }

        # Detect every other frame for performance
        detections = self.detect(frame) if self.frame_count % 2 == 0 else []

        tracked = self.tracker.update(detections)

        # Scene classification every 30 frames
        if self.frame_count % 30 == 0:
            self.classify_scene(frame)

        # Landing zone check every 15 frames
        landing_safe, landing_reason = True, 'not_checked'
        if self.frame_count % 15 == 0:
            landing_safe, landing_reason = self.check_landing_zone(frame)

        return {
            'detections': detections,
            'tracked': tracked,
            'scene': self._last_scene,
            'landing_safe': landing_safe,
            'landing_reason': landing_reason,
            'fps': round(self.fps, 1),
            'frame': frame,
        }

    def get_subject_position(self, tracked_objects, target_class='person'):
        """
        Find primary subject in tracked objects.
        Returns (cx, cy, area, track_id) of the largest matching object, or None.
        """
        best = None
        best_area = 0

        for tid, det in tracked_objects.items():
            if det.class_name == target_class or target_class == 'any':
                if det.area > best_area:
                    best_area = det.area
                    best = (det.center[0], det.center[1], det.area, tid)

        return best
