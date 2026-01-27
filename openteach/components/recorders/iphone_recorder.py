import os
import time
import cv2
import numpy as np
import h5py

from .recorder import Recorder
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraSubscriber, ZMQKeypointSubscriber
from openteach.utils.files import store_pickle_data
from openteach.constants import (
    IPHONE_CAM_INDEX,
    IPHONE_CAM_FPS,
    IPHONE_DEPTH_RESOLUTION,
)

class IPhoneCameraRecorder(Recorder):
    """
    Records the iPhone iphone RGB + Depth + Pose streams.

    Outputs (per demo):
      iphone_rgb.avi
      iphone_rgb.metadata

      iphone_depth.avi
      iphone_depth_images/*.bin
      iphone_depth.metadata

      iphone_pose.h5
    """

    def __init__(
        self,
        host: str,
        rgb_stream_port: int,
        depth_stream_port: int,
        pose_stream_port: int,
        storage_path: str,
        fps: int = None,
    ):
        self.notify_component_start(f"Recorder for cam_{IPHONE_CAM_INDEX}")

        self._host = host
        self._rgb_port = rgb_stream_port
        self._depth_port = depth_stream_port
        self._pose_port = pose_stream_port

        self._storage = storage_path
        os.makedirs(self._storage, exist_ok=True)

        self._fps = fps or IPHONE_CAM_FPS
        self.timer = FrequencyTimer(self._fps)

        self.rgb_sub = ZMQCameraSubscriber(host=host, port=rgb_stream_port, topic_type="RGB")
        self.depth_sub = ZMQCameraSubscriber(host=host, port=depth_stream_port, topic_type="Depth")
        self.pose_sub = ZMQKeypointSubscriber(host=host, port=pose_stream_port, topic="cam_60_pose")

        self.rgb_path = os.path.join(self._storage, "iphone_rgb_video.avi")
        self.depth_path = os.path.join(self._storage, "iphone_depth.avi")
        self.pose_h5_path = os.path.join(self._storage, "iphone_pose.h5")
        self.pose_txt_path = os.path.join(self._storage, "iphone_pose.txt")
        self.depth_bin_dir = os.path.join(self._storage, "iphone_depth_images")
        os.makedirs(self.depth_bin_dir, exist_ok=True)

        self._rgb_writer = None
        self._depth_writer = None

        self._depth_h, self._depth_w = IPHONE_DEPTH_RESOLUTION

        self.rgb_timestamps = []
        self.depth_timestamps = []
        self.pose_list = []

        self.rgb_frame_count = 0
        self.depth_frame_count = 0
        self.pose_frame_count = 0
        self.num_frames = 0

        self.sensor_name = f"cam_{IPHONE_CAM_INDEX}"
        self._recorder_file_name = os.path.join(self._storage, self.sensor_name)

    def _init_rgb_writer(self, frame_bgr):
        if self._rgb_writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._rgb_writer = cv2.VideoWriter(self.rgb_path, fourcc, self._fps, (w, h))
            print(f"[IPhoneRecorder] RGB writer initialized: {self.rgb_path}, size=({w}x{h}), fps={self._fps}")

    def _init_depth_writer(self):
        if self._depth_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._depth_writer = cv2.VideoWriter(
                self.depth_path, fourcc, self._fps, (self._depth_w, self._depth_h)
            )
            print(f"[IPhoneRecorder] Depth writer initialized: {self.depth_path}, size=({self._depth_w}x{self._depth_h}), fps={self._fps}")

    def _write_depth_visualization(self, depth_mm):
        if self._depth_writer is None:
            self._init_depth_writer()

        depth_m = depth_mm.astype(np.float32) * 0.001

        d_min = depth_m.min()
        d_max = depth_m.max()
        if d_max > d_min:
            d_norm = (depth_m - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(depth_m)

        d8 = (d_norm * 255).astype(np.uint8)
        d8_bgr = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
        self._depth_writer.write(d8_bgr)

    def stream(self):
        print(f"[IPhoneRecorder] Recording iphone from RGB:{self._rgb_port}, Depth:{self._depth_port}, Pose:{self._pose_port}")
        self.record_start_time = time.time()

        while True:
            try:
                self.timer.start_loop()

                try:
                    frame_bgr, ts_ns = self.rgb_sub.recv_rgb_image()
                    self._init_rgb_writer(frame_bgr)
                    self._rgb_writer.write(frame_bgr)
                    self.rgb_timestamps.append(ts_ns)
                    self.rgb_frame_count += 1
                except Exception:
                    pass

                try:
                    depth_mm, ts_ns = self.depth_sub.recv_depth_image()
                    self.depth_timestamps.append(ts_ns)
                    self.depth_frame_count += 1

                    bin_path = os.path.join(self.depth_bin_dir, f"frame_{self.depth_frame_count - 1:06d}.bin")
                    depth_mm.astype(np.float32).tofile(bin_path)

                    self._write_depth_visualization(depth_mm)
                except Exception:
                    pass

                try:
                    pose_dict = self.pose_sub.recv_keypoints()
                    print("[IPhoneRecorder] Received pose dict:", list(pose_dict.keys()))
                    ts = pose_dict["timestamp"]
                    t = np.array(pose_dict["t"], dtype=np.float32)
                    q = np.array(pose_dict["q"], dtype=np.float32)
                    self.pose_list.append((ts, t, q))
                    self.pose_frame_count += 1
                except Exception:
                    pass

                self.num_frames += 1
                self.timer.end_loop()

            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        print(f"[IPhoneRecorder] Recorded frames: RGB={self.rgb_frame_count}, Depth={self.depth_frame_count}, Pose={self.pose_frame_count}, Total={self.num_frames}")
        self._finalize_recording()

    def _write_rgb_metadata(self):
        path = os.path.join(self._storage, "iphone_rgb_video.metadata")
        meta = dict(self.metadata)
        meta["sensor"] = "iphone_rgb_video"
        meta["fps"] = self._fps
        meta["frame_count"] = self.rgb_frame_count
        meta["timestamps"] = self.rgb_timestamps
        meta["recorder_ip_address"] = self._host
        meta["recorder_image_stream_port"] = self._rgb_port
        store_pickle_data(path, meta)
        print(f"[IPhoneRecorder] RGB metadata saved to: {path}")

    def _write_depth_metadata(self):
        path = os.path.join(self._storage, "iphone_depth.metadata")
        meta = dict(self.metadata)
        meta["sensor"] = "iphone_depth"
        meta["fps"] = self._fps
        meta["frame_count"] = self.depth_frame_count
        meta["timestamps"] = self.depth_timestamps
        meta["recorder_ip_address"] = self._host
        meta["recorder_image_stream_port"] = self._depth_port
        store_pickle_data(path, meta)
        print(f"[IPhoneRecorder] Depth metadata saved to: {path}")

    def _write_pose_txt(self):
        txt_path = os.path.join(self._storage, "iphone_pose.txt")

        with open(txt_path, "w") as f:
            for ts, t, q in self.pose_list:
                q_list = q.tolist()
                t_list = t.tolist()
                line = f"\"<{ts}>\" ," + ",".join([str(v) for v in q_list + t_list]) + "\n"
                f.write(line)

        print(f"[IPhoneRecorder] Pose TXT saved: {txt_path}")

    def _write_pose_h5(self):
        print(f"[IPhoneRecorder] Saving pose data to: {self.pose_h5_path}")

        with h5py.File(self.pose_h5_path, "w") as f:
            if len(self.pose_list) == 0:
                f.create_dataset("timestamps", data=np.array([], dtype=np.float64))
                f.create_dataset("translations", data=np.zeros((0, 3), dtype=np.float32))
                f.create_dataset("quaternions", data=np.zeros((0, 4), dtype=np.float32))
                f.update(self.metadata)
                print("[IPhoneRecorder] No pose samples recorded; empty pose file created.")
                return

            timestamps = []
            translations = []
            quaternions = []

            for ts, t, q in self.pose_list:
                timestamps.append(ts)
                translations.append(t)
                quaternions.append(q)

            timestamps = np.array(timestamps, dtype=np.float64)
            translations = np.array(translations, dtype=np.float32)
            quaternions = np.array(quaternions, dtype=np.float32)

            f.create_dataset("timestamps", data=timestamps, compression="gzip", compression_opts=6)
            f.create_dataset("translations", data=translations, compression="gzip", compression_opts=6)
            f.create_dataset("quaternions", data=quaternions, compression="gzip", compression_opts=6)
            f.update(self.metadata)

        print(f"[IPhoneRecorder] Pose data saved in {self.pose_h5_path}")

    def _finalize_recording(self):
        for sub in [self.rgb_sub, self.depth_sub, self.pose_sub]:
            try:
                sub.stop()
            except Exception:
                pass

        for writer in [self._rgb_writer, self._depth_writer]:
            try:
                if writer:
                    writer.release()
            except Exception:
                pass

        self._display_statistics(self.num_frames)
        self._add_metadata(self.num_frames)

        self._write_rgb_metadata()
        self._write_depth_metadata()
        # self._write_pose_h5()
        self._write_pose_txt()

        def report(path):
            if os.path.exists(path):
                sz = os.path.getsize(path) / (1024 * 1024)
                print(f"[Done] {path} ({sz:.2f} MB)")
            else:
                print(f"[Done] {path} ✗ not found")

        report(self.rgb_path)
        report(self.depth_path)
        print(f"[Done] {self.depth_bin_dir}/ ({len(os.listdir(self.depth_bin_dir)) if os.path.exists(self.depth_bin_dir) else 0} depth .bin frames)")
        report(self.pose_txt_path)
