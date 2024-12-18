import cv2
import numpy as np
from matplotlib import pyplot as plt

# Video path
path1 = "/home/josten/Desktop/OpenCvRoisten/road.mp4"
cap = cv2.VideoCapture(path1)

if not cap.isOpened():
    print("no ei ole videot ju.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (600, 600))
    blur = cv2.GaussianBlur(resized_frame, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    height, width = resized_frame.shape[:2]

    
    src_points = np.float32([
        [width * 0.02, height * 0.85],
        [width * 0.99, height * 0.74],
        [width * 0.8, height * 0.68],
        [width * 0.34, height * 0.68]
    ])
    dst_points = np.float32([
        [0, height],
        [width, height],
        [width, 0],
        [0, 0]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    birdseye_view = cv2.warpPerspective(gray, matrix, (width, height))

   
    canny = cv2.Canny(birdseye_view, 20, 40)

    midpoint = width // 2
    right_x = np.where(canny[:, midpoint:] > 0)[1] + midpoint  
    right_y = np.where(canny[:, midpoint:] > 0)[0]

    nwindows = 6
    window_height = height // nwindows
    right_points = []
    mid_points = []

    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        right_inds = (right_y >= win_y_low) & (right_y < win_y_high)
        if np.sum(right_inds) > 0:
            right_x_center = np.mean(right_x[right_inds])
            right_y_center = np.mean(right_y[right_inds])
            right_points.append((int(right_x_center), int(right_y_center)))

        if np.sum(right_inds) > 0:
            mid_x = right_x_center
            mid_y = right_y_center
            mid_points.append((int(mid_x), int(mid_y)))

    print("\nVektorid järjestikuste keskpunktide vahel:")
    for i in range(1, len(mid_points)):
        x1, y1 = mid_points[i - 1]
        x2, y2 = mid_points[i]


        vector_x = x2 - x1
        vector_y = y2 - y1

        angle_radians = np.arctan2(vector_x, vector_y)  
        angle_degrees = np.degrees(angle_radians)  

        if angle_degrees < 0:
            angle_degrees += 360

        if angle_degrees > 90:
            angle_degrees -= 180  

        print(f"Vektor {i} -> {i + 1}: ({vector_x}, {vector_y}), Nurk: {angle_degrees:.2f} kraadi")

    result_frame = cv2.cvtColor(birdseye_view, cv2.COLOR_GRAY2BGR)

    for point in right_points:
        cv2.circle(result_frame, point, 5, (0, 0, 255), -1)

    for i in range(1, len(mid_points)):
        start_point = mid_points[i - 1]
        end_point = mid_points[i]
        cv2.arrowedLine(result_frame, start_point, end_point, (255, 0, 0), 2, tipLength=0.05)

    cv2.imshow("Window", resized_frame)
    cv2.imshow("Midpoint", result_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
