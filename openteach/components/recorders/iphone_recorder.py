import os
import time
import cv2
import numpy as np
import subprocess

from .recorder import Recorder
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraSubscriber, ZMQKeypointSubscriber
from openteach.constants import (
    IPHONE_CAM_INDEX,
    IPHONE_CAM_FPS,
    IPHONE_DEPTH_RESOLUTION,
)

class IPhoneCameraRecorder(Recorder):
    """
    Records the iPhone cam_60 RGB + Depth + Pose streams.
    Outputs:
      cam_60_rgb_video.mp4
      cam_60_depth.mp4
      cam_60_depth_images/*.bin
      cam_60_pose.txt
    """

    def __init__(
        self,
        host: str,
        rgb_stream_port: int,
        depth_stream_port: int,
        pose_stream_port: int,
        storage_path: str,
        fps: int = None,
        depth_vis_min_m: float = 0.2,
        depth_vis_max_m: float = 5.0,
    ):
        self.notify_component_start(f"Recorder for cam_{IPHONE_CAM_INDEX}")

        self._host = host
        self._rgb_port = rgb_stream_port
        self._depth_port = depth_stream_port
        self._pose_port = pose_stream_port

        self._storage = storage_path
        os.makedirs(self._storage, exist_ok=True)

        # Timer
        self._fps = fps or IPHONE_CAM_FPS
        self.timer = FrequencyTimer(self._fps)

        # Subscribers
        self.rgb_sub = ZMQCameraSubscriber(host=host, port=rgb_stream_port, topic_type="RGB")
        self.depth_sub = ZMQCameraSubscriber(host=host, port=depth_stream_port, topic_type="Depth")
        self.pose_sub = ZMQKeypointSubscriber(host=host, port=pose_stream_port, topic="cam_60_pose")

        # Output paths
        self.rgb_path = os.path.join(self._storage, "cam_60_rgb_video.mp4")
        self.depth_path = os.path.join(self._storage, "cam_60_depth.mp4")
        self.depth_bin_dir = os.path.join(self._storage, "cam_60_depth_images")
        self.pose_path = os.path.join(self._storage, "cam_60_pose.txt")
        os.makedirs(self.depth_bin_dir, exist_ok=True)

        # Writers & buffers
        self._rgb_writer = None
        self._depth_writer = None
        self._depth_writer_failed = False
        self._depth_frames_buffer = []

        # Data tracking
        self.num_frames = 0
        self.pose_entries = []

        # Metadata
        self.sensor_name = f"cam_{IPHONE_CAM_INDEX}"
        self._recorder_file_name = os.path.join(self._storage, self.sensor_name)

        # Depth configuration
        self._depth_h, self._depth_w = IPHONE_DEPTH_RESOLUTION  # expected (256,192)
        self._vis_min = depth_vis_min_m
        self._vis_max = depth_vis_max_m

    # ----------------------------------------------------------------------

    def _init_rgb_writer(self, frame_bgr):
        if self._rgb_writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._rgb_writer = cv2.VideoWriter(self.rgb_path, fourcc, self._fps, (w, h))
            print(f"[IPhoneRecorder] RGB writer initialized: {self.rgb_path}, size=({w}x{h}), fps={self._fps}")

    def _init_depth_writer(self):
        """Initialize depth writer once at the start. No re-initialization per frame."""
        if self._depth_writer is not None:
            return  # Already initialized
        
        print(f"[IPhoneRecorder] DEBUG: _init_depth_writer called")
        print(f"[IPhoneRecorder] DEBUG: self._depth_h={self._depth_h}, self._depth_w={self._depth_w}")
        print(f"[IPhoneRecorder] DEBUG: IPHONE_DEPTH_RESOLUTION={IPHONE_DEPTH_RESOLUTION}")
        
        try:
            # Try to create OpenCV VideoWriter with mp4v codec
            # VideoWriter expects (width, height) = (_depth_w, _depth_h) = (192, 256)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._depth_writer = cv2.VideoWriter(
                self.depth_path, fourcc, self._fps, (self._depth_w, self._depth_h)
            )
             
            if self._depth_writer.isOpened():
                print(f"[IPhoneRecorder] Depth writer initialized (OpenCV): {self.depth_path}, "
                      f"size=({self._depth_w}x{self._depth_h}), fps={self._fps}")
            else:
                print(f"[IPhoneRecorder] Depth writer failed to open; will use ffmpeg fallback")
                self._depth_writer = None
                self._depth_writer_failed = True
        except Exception as e:
            print(f"[IPhoneRecorder] Exception initializing depth writer: {e}")
            import traceback
            traceback.print_exc()
            self._depth_writer = None
            self._depth_writer_failed = True

    # ----------------------------------------------------------------------

    def _write_depth_visualization(self, depth_m):
        """
        Write depth frame to MP4 video.
        
        Depth frames arrive as float32 meters in shape (256, 192).
        Normalize to [0, 255] 8-bit grayscale, convert to BGR, write to video.
        If OpenCV writer fails, buffer for ffmpeg fallback.
        """
        # Initialize writer once on first frame
        if self._depth_writer is None and not self._depth_writer_failed:
            self._init_depth_writer()
        
        try:
            # Ensure depth is float32
            if depth_m.dtype != np.float32:
                depth_m = depth_m.astype(np.float32)
            
            # Incoming shape is (256, 192) = (height, width)
            # Resize if needed to match expected dimensions
            if depth_m.shape != (self._depth_h, self._depth_w):
                print(f"[IPhoneRecorder] Resizing depth from {depth_m.shape} to ({self._depth_h}, {self._depth_w})")
                depth_m = cv2.resize(depth_m, (self._depth_w, self._depth_h), interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0, 1] based on visualization min/max
            mn, mx = self._vis_min, self._vis_max
            d_norm = np.clip((depth_m - mn) / (mx - mn + 1e-6), 0, 1)
            d8 = (d_norm * 255).astype(np.uint8)
            
            # Convert grayscale to BGR
            d8_bgr = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
            
            # Write to OpenCV writer if available
            if self._depth_writer is not None and self._depth_writer.isOpened():
                # Note: cv2.VideoWriter.write() returns None, not a boolean
                # Do NOT check the return value or use it for re-initialization
                self._depth_writer.write(d8_bgr)
            else:
                # Fall back to buffering for ffmpeg
                if not self._depth_writer_failed:
                    print("[IPhoneRecorder] OpenCV writer not available; switching to ffmpeg fallback")
                    self._depth_writer_failed = True
                # Pass the normalized BGR frame to fallback
                self._write_depth_frame_fallback_bgr(d8_bgr)
                
        except Exception as e:
            print(f"[IPhoneRecorder] Exception in _write_depth_visualization: {e}")
            import traceback
            traceback.print_exc()
            self._depth_writer_failed = True

    def _write_depth_frame_fallback_bgr(self, d8_bgr):
        """Buffer already-normalized BGR frame for ffmpeg encoding fallback."""
        try:
            if not hasattr(self, '_fallback_logged'):
                print(f"[IPhoneRecorder] BUFFER: Starting to buffer frames (first frame shape={d8_bgr.shape}, dtype={d8_bgr.dtype})")
                self._fallback_logged = True
                
            self._depth_frames_buffer.append(d8_bgr.copy())
            
            # Log progress
            if len(self._depth_frames_buffer) % 30 == 0:
                print(f"[IPhoneRecorder] BUFFER: Buffered {len(self._depth_frames_buffer)} frames...")
                
        except Exception as e:
            print(f"[IPhoneRecorder] Exception in fallback buffering: {e}")
            import traceback
            traceback.print_exc()

    # ----------------------------------------------------------------------

    def stream(self):
        print(f"Recording cam_60 from RGB:{self._rgb_port}, Depth:{self._depth_port}, Pose:{self._pose_port}")
        self.record_start_time = time.time()
        
        depth_frame_count = 0
        rgb_frame_count = 0
        pose_frame_count = 0

        while True:
            try:
                self.timer.start_loop()

                # RGB
                try:
                    frame_bgr, ts_ns = self.rgb_sub.recv_rgb_image()
                    self._init_rgb_writer(frame_bgr)
                    self._rgb_writer.write(frame_bgr)
                    rgb_frame_count += 1
                except Exception as e:
                    pass  # Normal if no frames available

                # Depth
                try:
                    depth_mm, _ = self.depth_sub.recv_depth_image()
                    depth_m = depth_mm.astype(np.float32) / 1000.0
                    bin_path = os.path.join(self.depth_bin_dir, f"frame_{self.num_frames:06d}.bin")
                    depth_m.tofile(bin_path)
                    self._write_depth_visualization(depth_m)
                    depth_frame_count += 1
                except Exception as e:
                    if depth_frame_count == 0 and self.num_frames < 10:
                        print(f"[IPhoneRecorder] Depth receive error (frame {self.num_frames}): {e}")

                # Pose
                try:
                    pose_dict = self.pose_sub.recv_keypoints()
                    ts = pose_dict["timestamp"]
                    t, q = pose_dict["t"], pose_dict["q"]
                    self.pose_entries.append(f"{ts} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
                    pose_frame_count += 1
                except Exception as e:
                    pass  # Normal if no frames available

                self.num_frames += 1
                self.timer.end_loop()
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        print(f"[IPhoneRecorder] Recorded frames: RGB={rgb_frame_count}, Depth={depth_frame_count}, Pose={pose_frame_count}, Total={self.num_frames}")
        self._finalize_recording()

    # ----------------------------------------------------------------------

    def _encode_depth_video_ffmpeg(self):
        """Encode buffered depth frames to MP4 using ffmpeg."""
        if not self._depth_frames_buffer:
            print("[IPhoneRecorder] No depth frames to encode via ffmpeg.")
            return False

        print(f"[IPhoneRecorder] FFMPEG: Encoding {len(self._depth_frames_buffer)} depth frames")
        print(f"[IPhoneRecorder] FFMPEG: Frame 0 shape={self._depth_frames_buffer[0].shape}, dtype={self._depth_frames_buffer[0].dtype}, min={self._depth_frames_buffer[0].min()}, max={self._depth_frames_buffer[0].max()}")
        
        # CRITICAL: Frame is (height, width, 3) = (256, 192, 3)
        # But ffmpeg expects frames matching -video_size WxH = 192x256
        # So ffmpeg expects frames of size 192 (width) x 256 (height)
        frame_h, frame_w = self._depth_frames_buffer[0].shape[:2]
        print(f"[IPhoneRecorder] FFMPEG: Frame dimensions: {frame_w}x{frame_h} (WxH)")
        print(f"[IPhoneRecorder] FFMPEG: Video size param: {self._depth_w}x{self._depth_h}")
        
        if frame_w != self._depth_w or frame_h != self._depth_h:
            print(f"[IPhoneRecorder] ERROR: Frame size mismatch! Frames are {frame_w}x{frame_h} but ffmpeg expects {self._depth_w}x{self._depth_h}")
            return False
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{self._depth_w}x{self._depth_h}",
                "-framerate", str(self._fps),
                "-i", "pipe:",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                self.depth_path
            ]
            print(f"[IPhoneRecorder] FFMPEG: Command: {' '.join(cmd)}")
            
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            bytes_written = 0
            frame_bytes_per_frame = self._depth_w * self._depth_h * 3  # BGR24
            print(f"[IPhoneRecorder] FFMPEG: Expected bytes per frame: {frame_bytes_per_frame}")
            
            for i, frame in enumerate(self._depth_frames_buffer):
                try:
                    frame_bytes = frame.tobytes()
                    if len(frame_bytes) != frame_bytes_per_frame:
                        print(f"[IPhoneRecorder] ERROR: Frame {i} has {len(frame_bytes)} bytes, expected {frame_bytes_per_frame}")
                        
                    proc.stdin.write(frame_bytes)
                    bytes_written += len(frame_bytes)
                except BrokenPipeError:
                    print(f"[IPhoneRecorder] ERROR: ffmpeg process closed at frame {i}")
                    break
                except Exception as e:
                    print(f"[IPhoneRecorder] ERROR writing frame {i}: {e}")
                    break
                    
                if (i + 1) % 30 == 0:
                    print(f"[IPhoneRecorder] FFMPEG: Wrote {i+1}/{len(self._depth_frames_buffer)} frames ({bytes_written / (1024*1024):.1f} MB)...")
            
            print(f"[IPhoneRecorder] FFMPEG: Total bytes written to stdin: {bytes_written / (1024*1024):.1f} MB")
            
            try:
                proc.stdin.close()
            except Exception as e:
                print(f"[IPhoneRecorder] FFMPEG: Error closing stdin: {e}")
            
            # Wait for process with timeout
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print(f"[IPhoneRecorder] ERROR: ffmpeg timeout (>60s)")
                proc.kill()
                return False
            
            # Check return code
            if proc.returncode == 0:
                print("[IPhoneRecorder] ✓ ffmpeg depth encoding successful.")
                if os.path.exists(self.depth_path):
                    sz = os.path.getsize(self.depth_path) / (1024*1024)
                    print(f"[IPhoneRecorder] ✓ Output file size: {sz:.2f} MB")
                return True
            else:
                print(f"[IPhoneRecorder] ✗ ffmpeg failed (return code {proc.returncode})")
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                if stderr:
                    print(f"[IPhoneRecorder] FFMPEG stderr:\n{stderr}")
                return False
                
        except FileNotFoundError:
            print("[IPhoneRecorder] ERROR: ffmpeg not found. Install with: sudo apt-get install ffmpeg")
            return False
        except Exception as e:
            print(f"[IPhoneRecorder] Exception during ffmpeg encoding: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ----------------------------------------------------------------------

    def _finalize_recording(self):
        """Cleanup and metadata."""
        print(f"[IPhoneRecorder] _finalize_recording: _depth_writer_failed={self._depth_writer_failed}, buffer_size={len(self._depth_frames_buffer)}")
        
        for sub in [self.rgb_sub, self.depth_sub, self.pose_sub]:
            try:
                sub.stop()
            except Exception:
                pass

        for writer in [self._rgb_writer, self._depth_writer]:
            try:
                if writer: writer.release()
            except Exception:
                pass

        # Fallback encoding
        if self._depth_writer_failed and self._depth_frames_buffer:
            print("[IPhoneRecorder] Depth writer failed; attempting ffmpeg fallback...")
            self._encode_depth_video_ffmpeg()
        elif self._depth_writer_failed and not self._depth_frames_buffer:
            print("[IPhoneRecorder] WARNING: Depth writer failed but no frames buffered!")
        elif not self._depth_writer_failed and self._depth_frames_buffer:
            print(f"[IPhoneRecorder] WARNING: Depth writer succeeded but {len(self._depth_frames_buffer)} frames are in buffer!")

        # Write poses
        with open(self.pose_path, "w") as f:
            f.writelines(self.pose_entries)

        # Stats + metadata
        self._display_statistics(self.num_frames)
        self._add_metadata(self.num_frames)

        # Report saved files
        def report(path):
            if os.path.exists(path):
                sz = os.path.getsize(path) / (1024 * 1024)
                print(f"[Done] {path} ({sz:.2f} MB)")
            else:
                print(f"[Done] {path} ✗ not found")

        report(self.rgb_path)
        report(self.depth_path)
        print(f"[Done] {self.depth_bin_dir}/ ({len(os.listdir(self.depth_bin_dir)) if os.path.exists(self.depth_bin_dir) else 0} frames)")
        report(self.pose_path)
