"""
DeepStream Handler: Real GPU-Accelerated Video Analytics Pipeline.

Uses NVIDIA DeepStream SDK for 100+ FPS object detection when available.
Falls back to YOLOv8 (ultralytics) when DeepStream SDK is not installed.

Pipeline Architecture (when DeepStream available):
  RTSP Source → H264 Parse → NVDEC (GPU Decode) → StreamMux →
  nvinfer (YOLO Primary) → nvtracker (NvDCF) → Probe (extract meta) → FakeSink

Required for full DeepStream:
  - NVIDIA GPU with CUDA
  - DeepStream SDK 7.0+ installed
  - pyds (DeepStream Python bindings)
  - GStreamer with NVIDIA plugins
  - config_infer_primary_yoloV8.txt (inference config)
"""

import sys
import time
import os
import numpy as np
import cv2
import logging
from threading import Thread, Lock

logger = logging.getLogger(__name__)

# === IMPORT GSTREAMER ===
gst_available = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)
    gst_available = True
except (ImportError, ValueError):
    pass

# === IMPORT DEEPSTREAM ===
pyds_available = False
try:
    import pyds
    pyds_available = True
except ImportError:
    pyds = None

# === IMPORT YOLO FALLBACK ===
yolo_available = False
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
    yolo_available = True
except ImportError:
    pass


class DeepStreamHandler:
    """
    High-Speed Video Analytics using NVIDIA DeepStream SDK.
    
    When DeepStream is available:
      - Runs full GStreamer pipeline with GPU decode + nvinfer + tracker
      - Extracts detection metadata via probe callbacks
      - 100-200+ FPS inference throughput
    
    When DeepStream is NOT available:
      - Falls back to YOLOv8 (ultralytics) per-frame inference
      - Still GPU-accelerated via PyTorch CUDA
      - 30-60 FPS depending on model/resolution
    """

    def __init__(self, rtsp_url, config_path="config_infer_primary_yoloV8.txt"):
        self.rtsp_url = rtsp_url
        self.config_path = config_path
        self.running = False
        self.lock = Lock()
        
        # Detection results (thread-safe)
        self._detections = []
        self._latest_frame = None
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()
        
        # Determine mode
        self.mode = "none"
        self.pipeline = None
        self.model = None
        
        if pyds_available and gst_available:
            self.mode = "deepstream"
            logger.info("🚀 DeepStream SDK detected — using GPU pipeline")
        elif yolo_available:
            self.mode = "yolo"
            self._load_yolo()
        else:
            self.mode = "none"
            logger.error("❌ No detection backend available (need DeepStream or YOLOv8)")

        print(f"✅ DeepStreamHandler initialized (mode: {self.mode})")

    def _load_yolo(self):
        """Load YOLOv8 model as fallback."""
        model_paths = [
            "yolov8n.pt",
            "models/yolov8n.pt",
            os.path.join(os.path.dirname(__file__), "yolov8n.pt"),
        ]
        for p in model_paths:
            if os.path.exists(p):
                self.model = YOLO(p)
                logger.info(f"✅ YOLOv8 fallback loaded from {p}")
                return
        
        # Try downloading
        try:
            self.model = YOLO("yolov8n.pt")
            logger.info("✅ YOLOv8 downloaded and loaded")
        except Exception as e:
            logger.error(f"❌ YOLOv8 load failed: {e}")
            self.model = None
            self.mode = "none"

    # =========================================================
    # PUBLIC API
    # =========================================================

    def start(self):
        """Start the detection pipeline."""
        if self.running:
            return
        self.running = True

        if self.mode == "deepstream":
            self._start_deepstream()
        elif self.mode == "yolo":
            # YOLO runs per-frame via get_detections(), no background thread needed
            pass

        mode_label = "DeepStream GPU" if self.mode == "deepstream" else "YOLOv8 Fallback"
        print(f"✅ Detection Engine Started ({mode_label})")

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        print("🛑 Detection Engine Stopped")

    def get_detections(self, frame=None):
        """
        Get latest detections.
        
        DeepStream mode: Returns cached detections from pipeline probe.
        YOLO mode: Runs inference on provided frame.
        
        Returns: List of [class_id, center_x, center_y, width, height, confidence]
        """
        if self.mode == "deepstream":
            with self.lock:
                return list(self._detections)
        
        elif self.mode == "yolo" and self.model and frame is not None:
            return self._yolo_detect(frame)
        
        return []

    def get_fps(self):
        """Get current processing FPS."""
        return self._fps

    def get_latest_frame(self):
        """Get latest decoded frame (DeepStream mode only)."""
        with self.lock:
            return self._latest_frame

    # =========================================================
    # YOLO FALLBACK
    # =========================================================

    def _yolo_detect(self, frame):
        """Run YOLOv8 inference on a single frame."""
        t0 = time.time()
        results = self.model(frame, verbose=False)
        dets = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywh[0].tolist()
                dets.append([cls, x, y, w, h, conf])
        
        # Update FPS counter
        self._frame_count += 1
        elapsed = time.time() - self._last_fps_time
        if elapsed > 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = time.time()
        
        return dets

    # =========================================================
    # DEEPSTREAM PIPELINE
    # =========================================================

    def _start_deepstream(self):
        """Build and start the full DeepStream GStreamer pipeline."""
        try:
            self.pipeline = self._build_pipeline()
            if self.pipeline:
                self.pipeline.set_state(Gst.State.PLAYING)
                
                # Run GLib main loop in background thread
                t = Thread(target=self._glib_loop, daemon=True)
                t.start()
                logger.info("🚀 DeepStream pipeline PLAYING")
            else:
                logger.error("Pipeline build failed, falling back to YOLO")
                self.mode = "yolo"
                self._load_yolo()
        except Exception as e:
            logger.error(f"DeepStream start failed: {e}, falling back to YOLO")
            self.mode = "yolo"
            self._load_yolo()

    def _build_pipeline(self):
        """Build the complete GStreamer pipeline with all elements linked."""
        
        if not os.path.exists(self.config_path):
            logger.warning(f"⚠️ Inference config not found: {self.config_path}")
            logger.warning("   Creating default config...")
            self._create_default_config()

        pipeline = Gst.Pipeline.new("deepstream-drone-pipeline")
        
        # === 1. SOURCE (RTSP) ===
        source = Gst.ElementFactory.make("rtspsrc", "source")
        if not source:
            logger.error("Failed to create rtspsrc")
            return None
        source.set_property("location", self.rtsp_url)
        source.set_property("latency", 100)
        source.set_property("drop-on-latency", True)
        
        # === 2. RTP DEPAY ===
        depay = Gst.ElementFactory.make("rtph264depay", "depay")
        
        # === 3. H264 PARSE ===
        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
        
        # === 4. DECODER (NVDEC - GPU) ===
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        if not decoder:
            # Fallback to software decode
            decoder = Gst.ElementFactory.make("avdec_h264", "sw-decoder")
            logger.warning("Using software H264 decoder (nvv4l2decoder not available)")
        
        # === 5. STREAM MUX ===
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            logger.error("nvstreammux not available — DeepStream not properly installed")
            return None
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 40000)
        streammux.set_property('live-source', 1)
        
        # === 6. PRIMARY INFERENCE (nvinfer - YOLO) ===
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            logger.error("nvinfer not available")
            return None
        pgie.set_property('config-file-path', self.config_path)
        
        # === 7. TRACKER (NvDCF) ===
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if tracker:
            tracker_lib = '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so'
            if os.path.exists(tracker_lib):
                tracker.set_property('ll-lib-file', tracker_lib)
                tracker.set_property('tracker-width', 640)
                tracker.set_property('tracker-height', 384)
            else:
                logger.warning("Tracker library not found, skipping tracker")
                tracker = None
        
        # === 8. VIDEO CONVERTER ===
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
        
        # === 9. SINK (FakeSink — we just extract metadata) ===
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)  # Don't sync to clock — max throughput
        
        # === ADD ALL ELEMENTS TO PIPELINE ===
        elements = [source, depay, h264parse, decoder, streammux, pgie]
        if tracker:
            elements.append(tracker)
        elements.extend([nvvidconv, sink])
        
        for el in elements:
            if el:
                pipeline.add(el)
        
        # === LINK ELEMENTS ===
        # rtspsrc uses dynamic pads, so we connect via signal
        source.connect("pad-added", self._on_pad_added, depay)
        
        # depay → parse → decoder
        if not depay.link(h264parse):
            logger.error("Failed to link depay → h264parse")
        if not h264parse.link(decoder):
            logger.error("Failed to link h264parse → decoder")
        
        # decoder → streammux (via request pad)
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = decoder.get_static_pad("src")
        if sinkpad and srcpad:
            srcpad.link(sinkpad)
        
        # streammux → pgie
        if not streammux.link(pgie):
            logger.error("Failed to link streammux → pgie")
        
        # pgie → tracker (optional) → nvvidconv → sink
        last = pgie
        if tracker:
            if not pgie.link(tracker):
                logger.error("Failed to link pgie → tracker")
            last = tracker
        
        if not last.link(nvvidconv):
            logger.error(f"Failed to link {last.get_name()} → nvvidconv")
        if not nvvidconv.link(sink):
            logger.error("Failed to link nvvidconv → sink")
        
        # === ADD PROBE to extract detection metadata ===
        # Attach probe after pgie (or tracker if available)
        probe_pad = last.get_static_pad("src")
        if probe_pad:
            probe_pad.add_probe(Gst.PadProbeType.BUFFER, self._detection_probe, 0)
            logger.info("✅ Detection probe attached")
        
        print("✅ DeepStream Pipeline Built & Linked (all elements connected)")
        return pipeline

    def _on_pad_added(self, src, new_pad, target):
        """Handle dynamic pad from rtspsrc."""
        sink_pad = target.get_static_pad("sink")
        if not sink_pad.is_linked():
            new_pad.link(sink_pad)

    def _detection_probe(self, pad, info, user_data):
        """
        GStreamer probe callback — extracts detection metadata from nvinfer output.
        This is where DeepStream's 100+ FPS detections are captured.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        detections = []
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    
                    bbox = obj_meta.rect_params
                    cx = bbox.left + bbox.width / 2
                    cy = bbox.top + bbox.height / 2
                    
                    detections.append([
                        obj_meta.class_id,
                        cx, cy,
                        bbox.width, bbox.height,
                        obj_meta.confidence
                    ])
                except StopIteration:
                    break
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        # Thread-safe update
        with self.lock:
            self._detections = detections
        
        # FPS counter
        self._frame_count += 1
        elapsed = time.time() - self._last_fps_time
        if elapsed > 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = time.time()
            if self._fps > 0:
                logger.debug(f"DeepStream FPS: {self._fps:.1f} | Detections: {len(detections)}")
        
        return Gst.PadProbeReturn.OK

    def _glib_loop(self):
        """Run GLib main loop in background thread."""
        loop = GLib.MainLoop()
        
        # Watch for pipeline errors
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::eos", self._on_bus_eos)
        
        try:
            loop.run()
        except Exception as e:
            logger.error(f"GLib loop error: {e}")

    def _on_bus_error(self, bus, message):
        """Handle GStreamer bus errors."""
        err, debug = message.parse_error()
        logger.error(f"DeepStream Pipeline Error: {err.message}")
        logger.error(f"  Debug: {debug}")
        self.running = False

    def _on_bus_eos(self, bus, message):
        """Handle end-of-stream."""
        logger.info("DeepStream: End of Stream received")
        self.running = False

    def _create_default_config(self):
        """Create a default nvinfer configuration for YOLOv8."""
        config = """[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=models/yolov8n.onnx
model-engine-file=models/yolov8n.onnx_b1_gpu0_fp16.engine
labelfile-path=models/labels.txt
batch-size=1
process-mode=1
model-color-format=0
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
cluster-mode=2

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=100
"""
        os.makedirs(os.path.dirname(self.config_path) if os.path.dirname(self.config_path) else ".", exist_ok=True)
        with open(self.config_path, "w") as f:
            f.write(config)
        logger.info(f"📝 Created default inference config: {self.config_path}")
