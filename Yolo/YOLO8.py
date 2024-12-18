import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

def get_depth_at_center(depth_frame):
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    center_x = width // 2
    center_y = height // 2

    depth_image = np.asanyarray(depth_frame.get_data())
    center_depth = depth_image[center_y, center_x]

    depth_in_meters = center_depth / 1000.0

    return center_x, center_y, depth_in_meters

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        results = model.predict(color_image, conf=0.5)
        annotated_frame = results[0].plot()

        center_x, center_y, depth_at_center = get_depth_at_center(depth_frame)

        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

        cv2.putText(annotated_frame, f"Distance at center: {depth_at_center:.2f} meters",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 Real-time Detection with Distance Measurement (Intel RealSense)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
