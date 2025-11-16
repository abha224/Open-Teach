import os
import tempfile

from openteach.components.recorders.iphone_recorder import IPhoneCameraRecorder


def test_recorder_filename_base():
    tmpdir = tempfile.mkdtemp(prefix="iphone_rec_test_")
    try:
        rec = IPhoneCameraRecorder(
            host="127.0.0.1",
            rgb_stream_port=10100,
            depth_stream_port=10200,
            pose_stream_port=10400,
            storage_path=tmpdir,
            fps=30,
        )
        expected = os.path.join(tmpdir, rec.sensor_name)
        assert rec._recorder_file_name == expected, f"Expected {rec._recorder_file_name} to equal {expected}"
        print("OK: _recorder_file_name set to sensor_name base")
    finally:
        # cleanup: stop subscribers to close ZMQ sockets created in __init__
        try:
            if hasattr(rec, 'rgb_sub') and rec.rgb_sub is not None:
                rec.rgb_sub.stop()
        except Exception:
            pass
        try:
            if hasattr(rec, 'depth_sub') and rec.depth_sub is not None:
                rec.depth_sub.stop()
        except Exception:
            pass
        try:
            if hasattr(rec, 'pose_sub') and rec.pose_sub is not None:
                rec.pose_sub.stop()
        except Exception:
            pass
        # do not remove tmpdir automatically to avoid race with other processes


if __name__ == '__main__':
    test_recorder_filename_base()
