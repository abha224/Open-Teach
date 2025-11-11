import os
import time
import cv2
import numpy as np

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
    Output:
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
        self.timer = FrequencyTimer(fps or IPHONE_CAM_FPS)

        # Subscribers
        self.rgb_sub = ZMQCameraSubscriber(
            host=host, port=rgb_stream_port, topic_type="RGB"
        )
        self.depth_sub = ZMQCameraSubscriber(
            host=host, port=depth_stream_port, topic_type="Depth"
        )
        self.pose_sub = ZMQKeypointSubscriber(
            host=host, port=pose_stream_port, topic="cam_60_pose"
        )

        # Output files
        self.rgb_path = os.path.join(self._storage, "cam_60_rgb_video.mp4")
        self.depth_path = os.path.join(self._storage, "cam_60_depth.mp4")
        self.depth_bin_dir = os.path.join(self._storage, "cam_60_depth_images")
        self.pose_path = os.path.join(self._storage, "cam_60_pose.txt")
        os.makedirs(self.depth_bin_dir, exist_ok=True)

        self._rgb_writer = None
        self._depth_writer = None

        # Data counters
        self.num_frames = 0
        self.pose_entries = []

        self.sensor_name = f"cam_{IPHONE_CAM_INDEX}"
        self.record_start_time = None
        self.record_end_time = None

        self._depth_h, self._depth_w = IPHONE_DEPTH_RESOLUTION
        self._vis_min = depth_vis_min_m
        self._vis_max = depth_vis_max_m

    # ----------------------------------------------------------------------

    def _init_rgb_writer(self, frame_bgr):
        if self._rgb_writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._rgb_writer = cv2.VideoWriter(
                self.rgb_path, fourcc, self.timer.freq, (w, h)
            )

    def _init_depth_writer(self):
        if self._depth_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._depth_writer = cv2.VideoWriter(
                self.depth_path, fourcc, self.timer.freq, (self._depth_w, self._depth_h)
            )

    def _write_depth_visualization(self, depth_m):
        self._init_depth_writer()
        mn, mx = self._vis_min, self._vis_max
        d_norm = np.clip((depth_m - mn) / (mx - mn + 1e-6), 0, 1)
        d8 = (d_norm * 255).astype(np.uint8)
        d8_bgr = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
        self._depth_writer.write(d8_bgr)

    # ----------------------------------------------------------------------

    def stream(self):
        print(f"Recording cam_60 from RGB:{self._rgb_port}, Depth:{self._depth_port}, Pose:{self._pose_port}")
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()

                # RGB
                try:
                    frame_bgr, ts_ns = self.rgb_sub.recv_rgb_image()
                    self._init_rgb_writer(frame_bgr)
                    self._rgb_writer.write(frame_bgr)
                except Exception:
                    pass

                # Depth
                try:
                    depth_mm, _ = self.depth_sub.recv_depth_image()
                    depth_m = depth_mm.astype(np.float32) / 1000.0

                    bin_path = os.path.join(self.depth_bin_dir, f"frame_{self.num_frames:06d}.bin")
                    depth_m.astype(np.float32).tofile(bin_path)

                    self._write_depth_visualization(depth_m)
                except Exception:
                    pass

                # Pose
                try:
                    pose_dict = self.pose_sub.recv_keypoints()
                    ts = pose_dict["timestamp"]
                    t = pose_dict["t"]
                    q = pose_dict["q"]
                    self.pose_entries.append(f"{ts} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
                except Exception:
                    pass

                self.num_frames += 1
                self.timer.end_loop()

            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        self._finalize_recording()

    # ----------------------------------------------------------------------

    def _finalize_recording(self):
        # Close
        try: self.rgb_sub.stop()
        except: pass
        try: self.depth_sub.stop()
        except: pass
        try: self.pose_sub.stop()
        except: pass
        if self._rgb_writer: self._rgb_writer.release()
        if self._depth_writer: self._depth_writer.release()

        # Write poses
        with open(self.pose_path, "w") as f:
            f.writelines(self.pose_entries)

        # Stats + metadata
        self._display_statistics(self.num_frames)
        self._add_metadata(self.num_frames)

        print(f"[Done] {self.rgb_path}")
        print(f"[Done] {self.depth_path}")
        print(f"[Done] {self.depth_bin_dir}/")
        print(f"[Done] {self.pose_path}")
