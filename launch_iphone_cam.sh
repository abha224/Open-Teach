# Run in background with logging
mkdir -p iphone_logs
nohup python iphone_camera.py > iphone_logs/iphone_camera_log.txt 2>&1 &
