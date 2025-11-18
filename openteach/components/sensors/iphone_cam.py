import numpy as np
import time
import zmq
import json
from threading import Thread

from openteach.components import Component
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import (
    ZMQCameraPublisher,
    ZMQKeypointPublisher,
)
from openteach.constants import (
    IPHONE_CAM_INDEX,
    IPHONE_CAM_FPS,
)


class IPhoneUSBCamera(Component):
    """
    iPhone (cam_60) camera sensor publisher following RealSense API style.
    Expects streams: cam_60_rgb, cam_60_depth, cam_60_pose
    """

    def __init__(self, stream_configs, cam_id=IPHONE_CAM_INDEX):
        super().__init__()
        self.cam_id = cam_id

        self._host = stream_configs["host"]
        self._port = stream_configs["port"]

        # Setup publishers for Open-Teach pipeline
        self.rgb_publisher = ZMQCameraPublisher(
            host=self._host,
            port=self._port,
        )
        self.depth_publisher = ZMQCameraPublisher(
            host=self._host,
            port=self._port,
        )
        self.pose_publisher = ZMQKeypointPublisher(
            host=self._host,
            port=self._port + 99, # separate channel
        )

        self.timer = FrequencyTimer(IPHONE_CAM_FPS)

        # Latest data buffers
        self.rgb_frame = None      # (ts, image)
        self.depth_frame = None    # (ts, depth_f32_meters)
        self.pose_frame = None     # (ts, t(3), q(4))

        self._start_iphone_camera()

    def _make_sub(self, topic):
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(f"tcp://{self._host}:{self._port}")
        return sub

    def _start_iphone_camera(self):
        self.rgb_sub = self._make_sub("cam_60_rgb")
        self.depth_sub = self._make_sub("cam_60_depth")
        self.pose_sub = self._make_sub("cam_60_pose")

        self._listening = True
        Thread(target=self._listen_streams, daemon=True).start()

    def _listen_streams(self):
        while self._listening:
            self._receive_rgb()
            self._receive_depth()
            self._receive_pose()
            time.sleep(0.0005)

    def _receive_rgb(self):
        try:
            raw = self.rgb_sub.recv(flags=zmq.NOBLOCK)
            topic, payload = raw.split(b" ", 1)
            data = json.loads(payload.decode("utf-8"))
            rgb = np.frombuffer(
                bytes(data["rgb"]),
                dtype=np.uint8
            ).reshape(data["h"], data["w"], 3)
            self.rgb_frame = (data["ts_ns"], rgb)
        except zmq.Again:
            pass

    def _receive_depth(self):
        try:
            raw = self.depth_sub.recv(flags=zmq.NOBLOCK)
            topic, payload = raw.split(b" ", 1)
            data = json.loads(payload.decode("utf-8"))
            depth = np.frombuffer(
                bytes(data["depth"]),
                dtype=np.float32
            ).reshape(data["h"], data["w"])
            self.depth_frame = (data["ts_ns"], depth)
        except zmq.Again:
            pass

    def _receive_pose(self):
        try:
            raw = self.pose_sub.recv(flags=zmq.NOBLOCK)
            topic, payload = raw.split(b" ", 1)
            data = json.loads(payload.decode("utf-8"))
            t = np.array(data["t"], dtype=np.float32)
            q = np.array(data["q"], dtype=np.float32)
            self.pose_frame = (data["ts_ns"], t, q)
        except zmq.Again:
            pass

    def stream(self):
        self.notify_component_start(f"iPhone cam_{IPHONE_CAM_INDEX}")

        while True:
            try:
                self.timer.start_loop()

                if self.rgb_frame:
                    ts, img = self.rgb_frame
                    self.rgb_publisher.pub_rgb_image(img, ts)

                if self.depth_frame:
                    ts, depth_m = self.depth_frame
                    # depth_u16 = (depth_m * 1000).astype(np.uint16)
                    # self.depth_publisher.pub_depth_image(depth_u16, ts)
                    self.depth_publisher.pub_depth_image(
                        depth_m.astype(np.float32),
                        ts
                    )

                if self.pose_frame:
                    ts, t, q = self.pose_frame
                    payload = dict(
                        timestamp=ts,
                        t=t.tolist(),
                        q=q.tolist(),
                    )
                    self.pose_publisher.pub_keypoints(payload, topic_name="cam_60_pose")

                self.timer.end_loop()

            except KeyboardInterrupt:
                break

        self.shutdown()


    def shutdown(self):
        self._listening = False
        self.rgb_publisher.stop()
        self.depth_publisher.stop()
        self.pose_publisher.stop()
