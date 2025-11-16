#!/usr/bin/env python3
"""
Test depth video writer directly with synthetic frames.
Run this to diagnose if the issue is with VideoWriter or ffmpeg.
"""

import cv2
import numpy as np
import subprocess
import tempfile
import os

def test_opencv_writer():
    """Test OpenCV VideoWriter with synthetic depth frames."""
    print("\n" + "="*60)
    print("TEST 1: OpenCV VideoWriter")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_depth_opencv.mp4")
        fps = 30
        h, w = 256, 192  # Depth resolution
        
        print(f"Output: {output_path}")
        print(f"Resolution: {w}x{h} (width x height)")
        print(f"FPS: {fps}")
        
        # Try each codec
        for codec_str in ["mp4v", "XVID", "MJPG"]:
            try:
                print(f"\nTrying codec: {codec_str}")
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                
                if not writer.isOpened():
                    print(f"  ✗ Failed to open writer")
                    writer.release()
                    continue
                
                print(f"  ✓ Writer opened successfully")
                
                # Write 10 frames
                for i in range(10):
                    # Create synthetic depth frame
                    depth_m = np.random.uniform(0.2, 5.0, (h, w)).astype(np.float32)
                    d_norm = np.clip((depth_m - 0.2) / (5.0 - 0.2), 0, 1)
                    d8 = (d_norm * 255).astype(np.uint8)
                    d8_bgr = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
                    
                    success = writer.write(d8_bgr)
                    if not success:
                        print(f"  ✗ write() failed at frame {i}")
                        break
                    if i == 0:
                        print(f"  ✓ Frame 0 written successfully")
                
                writer.release()
                
                if os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024*1024)
                    if size_mb > 0.05:
                        print(f"  ✓ Output file created: {size_mb:.2f} MB")
                        print(f"  ✓✓✓ Codec '{codec_str}' WORKS! ✓✓✓")
                        return True
                    else:
                        print(f"  ✗ Output file too small: {size_mb:.2f} MB")
                else:
                    print(f"  ✗ Output file not created")
                    
            except Exception as e:
                print(f"  ✗ Exception: {e}")
    
    print("\n✗ No codec succeeded with OpenCV VideoWriter")
    return False

def test_ffmpeg_writer():
    """Test ffmpeg encoding with synthetic depth frames."""
    print("\n" + "="*60)
    print("TEST 2: ffmpeg Encoding")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_depth_ffmpeg.mp4")
        fps = 30
        h, w = 256, 192
        
        print(f"Output: {output_path}")
        print(f"Resolution: {w}x{h} (width x height)")
        print(f"FPS: {fps}")
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{w}x{h}",
                "-framerate", str(fps),
                "-i", "pipe:",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-loglevel", "quiet",
                output_path
            ]
            
            print(f"\nCommand: {' '.join(cmd[:10])}...\n")
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            # Write 10 synthetic frames
            bytes_written = 0
            for i in range(10):
                depth_m = np.random.uniform(0.2, 5.0, (h, w)).astype(np.float32)
                d_norm = np.clip((depth_m - 0.2) / (5.0 - 0.2), 0, 1)
                d8 = (d_norm * 255).astype(np.uint8)
                d8_bgr = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
                
                try:
                    frame_bytes = d8_bgr.tobytes()
                    proc.stdin.write(frame_bytes)
                    bytes_written += len(frame_bytes)
                except BrokenPipeError:
                    print(f"  ✗ Broken pipe at frame {i}")
                    break
                except Exception as e:
                    print(f"  ✗ Exception writing frame {i}: {e}")
                    break
                    
                if i == 0:
                    print(f"  ✓ Frame 0 written to ffmpeg")
            
            # Close stdin
            try:
                proc.stdin.close()
            except Exception as e:
                print(f"  Note: stdin close raised: {e}")
            
            # Wait for process
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print(f"  ✗ ffmpeg timeout")
                proc.kill()
                return False
            
            print(f"  ffmpeg return code: {proc.returncode}")
            
            if proc.returncode == 0:
                if os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024*1024)
                    if size_mb > 0.05:
                        print(f"  ✓ Output file created: {size_mb:.2f} MB")
                        print(f"  ✓✓✓ ffmpeg WORKS! ✓✓✓")
                        return True
                    else:
                        print(f"  ✗ Output file too small: {size_mb:.2f} MB")
                else:
                    print(f"  ✗ Output file not created")
            else:
                print(f"  ✗ ffmpeg failed")
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                if stderr:
                    print(f"  stderr (last 300 chars): {stderr[-300:]}")
        
        except FileNotFoundError:
            print("  ✗ ffmpeg not found. Install with: sudo apt-get install ffmpeg")
            return False
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("\nDEPTH VIDEO WRITER TEST")
    print("="*60)
    
    opencv_ok = test_opencv_writer()
    ffmpeg_ok = test_ffmpeg_writer()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"OpenCV VideoWriter: {'✓ OK' if opencv_ok else '✗ FAILED'}")
    print(f"ffmpeg encoding:    {'✓ OK' if ffmpeg_ok else '✗ FAILED'}")
    
    if not (opencv_ok or ffmpeg_ok):
        print("\n⚠ WARNING: Both methods failed! Check:")
        print("  1. Is ffmpeg installed? (sudo apt-get install ffmpeg)")
        print("  2. OpenCV codecs available? (python -c \"import cv2; print(cv2.getBuildInformation())\")")
        print("  3. Disk space available?")
    elif not opencv_ok and ffmpeg_ok:
        print("\n✓ Good: ffmpeg works as fallback for corrupted OpenCV VideoWriter")
    else:
        print("\n✓ OpenCV VideoWriter is available")
