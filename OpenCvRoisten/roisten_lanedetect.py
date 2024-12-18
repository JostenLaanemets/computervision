import cv2
import numpy as np

# Video path
path1 = "Test1.webm"
cap = cv2.VideoCapture(path1)

if not cap.isOpened():
    print("Video netu")
    exit()

# Trackbari jaoks
def nothing(x):
    pass

# Trackbaridega aknad
cv2.namedWindow("Settings")
cv2.createTrackbar("Canny Lower", "Settings", 20, 255, nothing)
cv2.createTrackbar("Canny Upper", "Settings", 40, 255, nothing)
cv2.createTrackbar("Gaussian Blur", "Settings", 5, 50, nothing)

cv2.createTrackbar("Yellow Lower H", "Settings", 16, 179, nothing)
cv2.createTrackbar("Yellow Upper H", "Settings", 30, 179, nothing)
cv2.createTrackbar("Yellow Lower S", "Settings", 70, 255, nothing)
cv2.createTrackbar("Yellow Upper S", "Settings", 255, 255, nothing)
cv2.createTrackbar("Yellow Lower V", "Settings", 160, 255, nothing)
cv2.createTrackbar("Yellow Upper V", "Settings", 255, 255, nothing)

# Video pilt
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (600, 600))
    height, width = resized_frame.shape[:2]

    # Trackbari väärtused
    canny_lower = cv2.getTrackbarPos("Canny Lower", "Settings")
    canny_upper = cv2.getTrackbarPos("Canny Upper", "Settings")
    blur_size = cv2.getTrackbarPos("Gaussian Blur", "Settings")
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  # Must be odd

    yellow_lower_h = cv2.getTrackbarPos("Yellow Lower H", "Settings")
    yellow_upper_h = cv2.getTrackbarPos("Yellow Upper H", "Settings")
    yellow_lower_s = cv2.getTrackbarPos("Yellow Lower S", "Settings")
    yellow_upper_s = cv2.getTrackbarPos("Yellow Upper S", "Settings")
    yellow_lower_v = cv2.getTrackbarPos("Yellow Lower V", "Settings")
    yellow_upper_v = cv2.getTrackbarPos("Yellow Upper V", "Settings")

    # Gaussian Blur
    blur = cv2.GaussianBlur(resized_frame, (blur_size, blur_size), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Yellow maskimine
    lower_yellow = np.array([yellow_lower_h, yellow_lower_s, yellow_lower_v])
    upper_yellow = np.array([yellow_upper_h, yellow_upper_s, yellow_upper_v])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Perspektiivimuundmine birdeye vaateks
    src_points = np.float32([
        [width * 0.18, height * 0.85],
        [width * 0.82, height * 0.85],
        [width * 0.7, height * 0.6],
        [width * 0.32, height * 0.6]
    ])
    dst_points = np.float32([
        [0, height],
        [width, height],
        [width, 0],
        [0, 0]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    birdseye_view = cv2.warpPerspective(yellow_mask, matrix, (width, height))

    # Histgramm
    # Keskpunkti leidmine
    midpoint = width // 2
    right_x = np.where(birdseye_view[:, midpoint:] > 0)[1] + midpoint
    right_y = np.where(birdseye_view[:, midpoint:] > 0)[0]

    # Joone punktide määramine
    points = 5
    window_height = height // points
    right_points = []

    for point in range(points):
        win_y_low = height - (point + 1) * window_height
        win_y_high = height - point * window_height

        # Parempoolne kasti piires olevad punktid
        right_inds = (right_y >= win_y_low) & (right_y < win_y_high)
        if np.sum(right_inds) > 0:
            right_x_center = np.mean(right_x[right_inds])
            right_y_center = np.mean(right_y[right_inds])
            right_points.append((int(right_x_center), int(right_y_center)))

    # offset joone punktidele
    offset = 200
    points_offset = [(point[0] - offset, point[1]) for point in right_points]

    # Birdseye view
    result_frame = cv2.cvtColor(birdseye_view, cv2.COLOR_GRAY2BGR)

    # Keskjoonte tegemine, mille järgi saame arvutada roboti asukohta
    left_line_x = 100
    right_line_x = 300
    cv2.line(result_frame, (left_line_x, 0), (left_line_x, height), (0, 255, 0), 2)  
    cv2.line(result_frame, (right_line_x, 0), (right_line_x, height), (0, 255, 0), 2)  
      
    center_x = (left_line_x + right_line_x) // 2
    cv2.line(result_frame, (center_x, 0), (center_x, height), (255, 0, 0), 2)

    # Votame 4nda punkti alguses
    if len(points_offset) > 3:
        center_point = points_offset[3]
        
        distance_from_center = center_point[0] - center_x  
        if left_line_x < center_point[0]<right_line_x:
            print(f"mid value: {distance_from_center} ") # Väärtus mis peab PID-i saatma


    # Punktide joonistamine 
    for i, point in enumerate(right_points):
        cv2.circle(result_frame, point, 5, (0, 0, 255), -1)  # Punane
        if i > 0:
            prev_point = right_points[i - 1]
            cv2.arrowedLine(result_frame, prev_point, point, (255, 0, 0), 3, tipLength=0.1)

    for i, point in enumerate(points_offset):
        if i == 3:
            cv2.circle(result_frame, point, 5, (0, 255, 0), -1) # Roheline
        else:
            cv2.circle(result_frame, point, 5, (255, 0, 0), -1)  # Sinine

    
    cv2.imshow("Window", resized_frame)
    cv2.imshow("Detected Yellow Lines", result_frame)
    #cv2.imshow("Yellow Mask", yellow_mask)
    

    # fps
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
