import cv2
import numpy as np

# Video tee
path1 = "Test1.webm"
cap = cv2.VideoCapture(path1)

if not cap.isOpened():
    print("error, no no no")
    exit()

# Trackbar-i jaoks funktsioon
def nothing(x):
    pass

# Loome akna trackbar-idega
cv2.namedWindow("Settings")
cv2.createTrackbar("Canny Lower", "Settings", 20, 255, nothing)
cv2.createTrackbar("Canny Upper", "Settings", 40, 255, nothing)
cv2.createTrackbar("Gaussian Blur", "Settings", 5, 50, nothing)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (600, 600))
    height, width = resized_frame.shape[:2]

    # Loeme trackbar väärtused
    canny_lower = cv2.getTrackbarPos("Canny Lower", "Settings")
    canny_upper = cv2.getTrackbarPos("Canny Upper", "Settings")
    blur_size = cv2.getTrackbarPos("Gaussian Blur", "Settings")
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  # Peab olema paaritu


    # Rakendame Gaussian Blur-i
    blur = cv2.GaussianBlur(resized_frame, (blur_size, blur_size), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Perspektiivimuundamine
    src_points = np.float32([
        [width * 0.18, height * 0.85],
        [width * 0.82, height * 0.85],
        [width * 0.7, height * 0.6],
        [width * 0.32, height * 0.6]
    ])
    dst_points = np.float32([
        [0    , height],
        [width, height],
        [width,      0],
        [0    ,      0]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    birdseye_view = cv2.warpPerspective(gray, matrix, (width, height))

    # Canny servatuvastus
    canny = cv2.Canny(birdseye_view, canny_lower, canny_upper)

    # ROI rakendamine
    roi = np.zeros_like(canny)
    polygon = np.array([
        [
            (width * 0.4, height),
            (width * 0.6, height),
            (width * 0.55, height * 0.6),
            (width * 0.45, height * 0.6)
            
        ]
    ], np.int32)
    for point in src_points:
        cv2.circle(resized_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)  # Green circles for source points

    cv2.fillPoly(roi, polygon, 255)
    canny = cv2.bitwise_and(canny, roi)

    # Histogrammi põhine jaotus
    midpoint = width // 2
    right_x = np.where(canny[:, midpoint:] > 0)[1] + midpoint
    right_y = np.where(canny[:, midpoint:] > 0)[0]

    
    nwindows = 6
    window_height = height // nwindows
    right_points = []

    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        # Parempoolne kasti piires olevad punktid
        right_inds = (right_y >= win_y_low) & (right_y < win_y_high)
        if np.sum(right_inds) > 0:
            right_x_center = np.mean(right_x[right_inds])
            right_y_center = np.mean(right_y[right_inds])
            right_points.append((int(right_x_center), int(right_y_center)))

    
    result_frame = cv2.cvtColor(birdseye_view, cv2.COLOR_GRAY2BGR)

    # Joonistame punktid
    for point in right_points:
        cv2.circle(result_frame, point, 5, (0, 0, 255), -1)

    # Joonistame vektorid 
    for i in range(1, len(right_points)):
        start_point = right_points[i - 1]
        end_point = right_points[i]
        cv2.arrowedLine(result_frame, start_point, end_point, (255, 0, 0), 2, tipLength=0.05)

   
    cv2.imshow("Window", resized_frame)
    cv2.imshow("Midpoint", result_frame)
    cv2.imshow("Canny", canny)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
