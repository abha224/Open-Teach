import os
import numpy as np

# Path to your depth folder
DEPTH_DIR = "/mnt/robotlab/abha/visk_rl_jax/exports/16_11_socket_aug/demonstration_304/cam_60_depth_images"

# ARKit depth resolution —
# Confirmed from your logs: (H=256, W=192)
H, W = 256, 192
FRAME_SIZE = H * W

all_files = sorted([f for f in os.listdir(DEPTH_DIR) if f.endswith(".bin")])

valid = 0
zero = 0
invalid_shape = 0

stats = []

print(f"Found {len(all_files)} depth .bin frames")

for fname in all_files:
    path = os.path.join(DEPTH_DIR, fname)
    
    depth = np.fromfile(path, dtype=np.float32)

    # Check total size
    if depth.size != FRAME_SIZE:
        print(f"[ERROR] {fname} wrong size: {depth.size} values instead of {FRAME_SIZE}")
        invalid_shape += 1
        continue

    depth = depth.reshape((H, W))

    mn, mx, mean, std = depth.min(), depth.max(), depth.mean(), depth.std()

    # Check if completely zero
    if mn == 0.0 and mx == 0.0:
        zero += 1
    else:
        valid += 1

    stats.append((fname, mn, mx, mean, std))

# Print summary
print("\n============================================================")
print("SUMMARY")
print("============================================================")
print(f"Total frames      : {len(all_files)}")
print(f"Valid depth frames: {valid}")
print(f"Zero frames       : {zero}")
print(f"Invalid shape     : {invalid_shape}")

if valid > 0:
    print("\nExample of a valid frame:")
    for s in stats:
        if s[1] != 0.0 or s[2] != 0.0:
            print(f"  {s[0]} → min={s[1]:.4f}, max={s[2]:.4f}, mean={s[3]:.4f}, std={s[4]:.4f}")
            break

if zero > 0:
    print("\nExample of a zero frame:")
    for s in stats:
        if s[1] == 0.0 and s[2] == 0.0:
            print(f"  {s[0]} → min={s[1]:.4f}, max={s[2]:.4f}, mean={s[3]:.4f}, std={s[4]:.4f}")
            break

print("============================================================\n")
