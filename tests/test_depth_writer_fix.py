#!/usr/bin/env python3
"""
Quick test to verify depth video writer produces valid MP4.
Synthetic depth data only; no external streams needed.
"""

import os
import sys
import tempfile
import cv2
import numpy as np

def test_depth_writer_with_codecs():
    """Test codec fallback logic and frame writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_depth.mp4")
        fps = 30
        depth_h, depth_w = 480, 640
        
        print(f"Testing depth writer in {tmpdir}")
        
        # Try each codec
        codecs = ["mp4v", "XVID", "MJPG"]
        writer = None
        used_codec = None
        
        for codec_str in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                test_writer = cv2.VideoWriter(
                    output_path, fourcc, fps, (depth_w, depth_h)
                )
                if test_writer.isOpened():
                    writer = test_writer
                    used_codec = codec_str
                    print(f"  ✓ Codec '{codec_str}' opened successfully")
                    break
                else:
                    print(f"  ✗ Codec '{codec_str}' failed to open")
                    test_writer.release()
            except Exception as e:
                print(f"  ✗ Codec '{codec_str}' error: {e}")
        
        if writer is None:
            print("  ✗ No codec succeeded. Test FAILED.")
            return False
        
        # Write 10 synthetic frames
        print(f"  Writing 10 frames with codec '{used_codec}'...")
        for i in range(10):
            # Create synthetic depth (0.2m to 5.0m range)
            depth_m = np.random.uniform(0.2, 5.0, (depth_h, depth_w)).astype(np.float32)
            
            # Normalize and convert to 8-bit BGR
            d_norm = np.clip((depth_m - 0.2) / (5.0 - 0.2), 0, 1)
            d8 = (d_norm * 255).astype(np.uint8)
            d8_bgr = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
            
            # Validate frame
            assert d8_bgr.shape == (depth_h, depth_w, 3), f"Frame shape mismatch: {d8_bgr.shape}"
            assert d8_bgr.dtype == np.uint8, f"Frame dtype mismatch: {d8_bgr.dtype}"
            
            success = writer.write(d8_bgr)
            if not success:
                print(f"    ✗ Frame {i} write failed")
                writer.release()
                return False
        
        writer.release()
        writer = None
        
        # Check output file
        if not os.path.exists(output_path):
            print(f"  ✗ Output file not created: {output_path}")
            return False
        
        size_bytes = os.path.getsize(output_path)
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"  ✓ Output file created: {size_mb:.2f} MB")
        
        if size_bytes < 1000:
            print(f"  ✗ Output file too small: {size_bytes} bytes (likely corrupt)")
            return False
        
        print("  ✓ Test PASSED")
        return True

if __name__ == "__main__":
    success = test_depth_writer_with_codecs()
    sys.exit(0 if success else 1)
