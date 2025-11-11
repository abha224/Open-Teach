import socket
import struct
import time
import numpy as np
import cv2
try:
    import lzfse
    _HAS_LZFSE = True
except Exception:
    _HAS_LZFSE = False
    print("Warning: LZFSE not available, depth stream will not work")

from openteach.components import Component
from openteach.utils.timer import FrequencyTimer
from openteach.utils.images import rescale_image
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from openteach.constants import VIZ_PORT_OFFSET, CAM_FPS

# Wire format (must match your iOS USBManager)
_PEERTALK_FMT   = ">IIII"        # PeerTalkHeader (a,b,c,body_size) big-endian
_RECORD3D_FMT   = "<IIIIIIIIII I"  # Record3DHeader little-endian
_INTR_FMT       = "<ffff"        # fx, fy, tx, ty
_POSE_FMT       = "<fffffff"     # qx,qy,qz,qw,tx,ty,tz

class IPhoneUSBCamera(Component):
    """
    USB-TCP client that ingests Record3D RGB + Depth (+ Confidence) and
    republishes to OpenTeach ZMQ publishers:
      - RGB  : cam_port_offset + cam_idx
      - Depth: cam_port_offset + cam_idx + DEPTH_PORT_OFFSET
      - Conf : (optional) conf_port_offset + cam_idx  [if provided in configs]
    Also sends a compressed RGB stream for Oculus viz on (set_port_offset + VIZ_PORT_OFFSET)
    when stream_oculus=True (same behavior as FishEye/RealSense).
    """
    def __init__(self, cam_idx, stream_configs, usb_tcp_host="127.0.0.1", usb_tcp_port=14500, stream_oculus=False):
        super().__init__()
        self.cam_idx = cam_idx
        self._stream_configs = stream_configs
        self._usb_host = usb_tcp_host
        self._usb_port = usb_tcp_port
        self._stream_oculus = stream_oculus

        # ZMQ publishers (match RealSense/Fisheye pattern)
        self.rgb_pub = ZMQCameraPublisher(
            host=stream_configs["host"],
            port=stream_configs["port"],  # cam_port_offset + cam_idx
        )
        self.depth_pub = ZMQCameraPublisher(
            host=stream_configs["host"],
            port=stream_configs["port"] + stream_configs["depth_port_add"],  # + DEPTH_PORT_OFFSET
        )

        # Optional confidence publisher if config provides a separate base
        self.conf_pub = None
        if "conf_port_add" in stream_configs:
            self.conf_pub = ZMQCameraPublisher(
                host=stream_configs["host"],
                port=stream_configs["port"] + stream_configs["conf_port_add"],  # e.g., conf offset family
            )

        # Optional compressed viz stream for Oculus (same style as FishEye)
        self.rgb_viz_pub = None
        if stream_oculus:
            self.rgb_viz_pub = ZMQCompressedImageTransmitter(
                host=stream_configs["host"],
                port=stream_configs["set_port_offset"] + VIZ_PORT_OFFSET,
            )

        self.timer = FrequencyTimer(CAM_FPS)
        self.sock = None

    # ---------- TCP helpers ----------
    def _connect_blocking(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self._usb_host, self._usb_port))
                print(f"[iPhoneUSB] Connected to {self._usb_host}:{self._usb_port}")
                return s
            except Exception as e:
                print(f"[iPhoneUSB] Waiting for iPhone USB stream… ({e})")
                time.sleep(0.5)

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket disconnected")
            buf += chunk
        return buf

    # ---------- main loop ----------
    def stream(self):
        self.notify_component_start('IPhoneUSBCamera')
        print(f"[iPhoneUSB] Starting stream on ZMQ {self._stream_configs['host']}:{self._stream_configs['port']} (cam_idx={self.cam_idx})")
        if self._stream_oculus:
            viz_port = self._stream_configs['set_port_offset'] + VIZ_PORT_OFFSET
            print(f"[iPhoneUSB] Oculus viz on port {viz_port}")

        self.sock = self._connect_blocking()

        while True:
            try:
                self.timer.start_loop()

                # 1) PeerTalk header
                pt = self._recv_exact(16)
                a, b, c, body_size = struct.unpack(_PEERTALK_FMT, pt)

                # 2) Body
                body = self._recv_exact(body_size)
                off = 0

                # Record3DHeader
                rec_sz = struct.calcsize(_RECORD3D_FMT)
                (rgbW, rgbH, dW, dH, confW, confH,
                 rgbSize, depthSize, confSize, misc, devType) = struct.unpack_from(_RECORD3D_FMT, body, off)
                off += rec_sz

                # Intrinsics (unused here)
                off += struct.calcsize(_INTR_FMT)
                # Pose (unused here)
                off += struct.calcsize(_POSE_FMT)

                # RGB (JPEG)
                rgb_jpeg = body[off: off + rgbSize]
                off += rgbSize
                rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, np.uint8), cv2.IMREAD_COLOR)

                # Depth
                depth = None
                if depthSize > 0 and _HAS_LZFSE:
                    depth_comp = body[off: off + depthSize]
                    off += depthSize
                    try:
                        depth_raw = lzfse.decompress(depth_comp)
                        depth = np.frombuffer(depth_raw, np.float32).reshape((dH, dW))
                    except Exception as e:
                        print(f"[iPhoneUSB] depth decode error: {e}")

                # Confidence
                conf = None
                if confSize > 0 and _HAS_LZFSE:
                    conf_comp = body[off: off + confSize]
                    off += confSize
                    try:
                        conf_raw = lzfse.decompress(conf_comp)
                        conf = np.frombuffer(conf_raw, np.uint8).reshape((confH, confW))
                    except Exception as e:
                        print(f"[iPhoneUSB] conf decode error: {e}")

                ts = time.time()

                # Publish: RGB
                if rgb is not None:
                    self.rgb_pub.pub_rgb_image(rgb, ts)
                    if self.rgb_viz_pub is not None:
                        self.rgb_viz_pub.send_image(rescale_image(rgb, 2))

                # Publish: Depth
                if depth is not None:
                    # OpenTeach depth publishers expect float32 arrays
                    self.depth_pub.pub_depth_image(depth, ts)

                # Publish: Confidence (optional)
                if self.conf_pub is not None and conf is not None:
                    # Publish as an 8-bit single-channel image
                    self.conf_pub.pub_rgb_image(conf, ts)

                self.timer.end_loop()

            except KeyboardInterrupt:
                break
            except ConnectionError:
                print("[iPhoneUSB] disconnected — reconnecting…")
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = self._connect_blocking()
            except Exception as e:
                print(f"[iPhoneUSB] runtime error: {e}")

        # cleanup
        if self.sock:
            try: self.sock.close()
            except: pass
        self.rgb_pub.stop()
        self.depth_pub.stop()
        if self.conf_pub: self.conf_pub.stop()
        if self.rgb_viz_pub: self.rgb_viz_pub.stop()
        print(f"[iPhoneUSB] shutdown for cam_idx={self.cam_idx}")