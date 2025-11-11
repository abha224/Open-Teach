import cv2
import numpy as np
from openteach.utils.network import ZMQCameraSubscriber
import time

def main():
    # Create subscribers for RGB and depth streams
    rgb_sub = ZMQCameraSubscriber(
        host='localhost',  # or your network IP
        port=10100,  # iphone_rgb_port_offset from network.yaml
        topic_type='RGB'
    )
    
    depth_sub = ZMQCameraSubscriber(
        host='localhost',  # or your network IP
        port=14500,  # iphone_depth_port_offset from network.yaml
        topic_type='Depth'
    )

    print("Waiting for frames from iPhone camera...")
    print("Press 'q' to exit")

    while True:
        try:
            # Get RGB frame
            rgb_frame, rgb_ts = rgb_sub.recv_rgb_image()
            if rgb_frame is not None:
                cv2.imshow('iPhone RGB', rgb_frame)

            # Get depth frame 
            depth_frame, depth_ts = depth_sub.recv_depth_image()
            if depth_frame is not None:
                # Normalize depth for visualization
                depth_viz = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imshow('iPhone Depth', depth_viz)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error receiving frames: {e}")
            time.sleep(0.1)

    # Cleanup
    cv2.destroyAllWindows()
    rgb_sub.stop()
    depth_sub.stop()

if __name__ == '__main__':
    main()