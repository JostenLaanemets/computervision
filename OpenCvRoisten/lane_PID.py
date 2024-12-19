import cv2
import numpy as np
import time
import rospy
from std_msgs.msg import Float32


# Video path
path1 = "/home/josten/Desktop/Cleveron/Rositen/OpenCvRoisten/Test1.webm"
cap = cv2.VideoCapture(path1)


# left_pub = rospy.Publisher('Left_vel', Float32, queue_size=10)
# right_pub = rospy.Publisher('Right_vel', Float32, queue_size=10)

lasttime = 0
#PID 
Kp= 0.4
Ki = 0.01
Kd = 0.05
error = 0.0
integral = 0.0
previous_error = 0.0
#dt = 1e-6

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

# rospy.init_node('Robot', anonymous=True)



while True:  #not rospy.is_shutdown():
    
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (600, 600))
        height, width = resized_frame.shape[:2]

        # Trackbari väärtused
        canny_lower = cv2.getTrackbarPos("Canny Lower", "Settings")
        canny_upper = cv2.getTrackbarPos("Canny Upper", "Settings")
        blur_size = cv2.getTrackbarPos("Gaussian Blur", "Settings")
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  # Gaussian Blur peab olema paaritu arv

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
        lower_yellow = np.array([yellow_lower_h, yellow_lower_s, yellow_lower_v]) # H, S, V
        upper_yellow = np.array([yellow_upper_h, yellow_upper_s, yellow_upper_v]) # H, S, V

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
        matrix = cv2.getPerspectiveTransform(src_points, dst_points) # perspektiivi muutmine
        birdseye_view = cv2.warpPerspective(yellow_mask, matrix, (width, height))

        # Histgramm
        # Keskpunkti leidmine
        midpoint = width // 2
        right_x = np.where(birdseye_view[:, midpoint:] > 0)[1] + midpoint # Parempoolne pool x 
        right_y = np.where(birdseye_view[:, midpoint:] > 0)[0]            # Parempoolne pool y

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
                right_x_center = np.mean(right_x[right_inds]) # keskpunkt
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
        cv2.line(result_frame, (left_line_x, 0), (left_line_x, height), (0, 255, 0), 2)  # Parempoolse joone joonistamine
        cv2.line(result_frame, (right_line_x, 0), (right_line_x, height), (0, 255, 0), 2)  # Vasakpoolse joone joonistamine
        
        center_x = (left_line_x + right_line_x) // 2
        cv2.line(result_frame, (center_x, 0), (center_x, height), (255, 0, 0), 2) # Keskpunkti joonistamine

        # Votame 4nda punkti alguses
        if len(points_offset) > 3:
            center_point = points_offset[3]
            
            distance_from_center = center_point[0] - center_x  

            if left_line_x < center_point[0]<right_line_x:
                error = distance_from_center
                #print(f"mid value: {distance_from_center} ") # Väärtus mis peab PID-i saatma


        # Punktide joonistamine 
        for i, point in enumerate(right_points):
            cv2.circle(result_frame, point, 5, (0, 0, 255), -1)  # Punane
            if i > 0:
                prev_point = right_points[i - 1]
                cv2.arrowedLine(result_frame, prev_point, point, (255, 0, 0), 3, tipLength=0.1)

        for i, point in enumerate(points_offset):
            if i == 3:
                cv2.circle(result_frame, point, 5, (0, 255, 0), -1) # Roheline
                cv2.putText(result_frame, f"({error})", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.circle(result_frame, point, 5, (255, 0, 0), -1)  # Sinine

        
        cv2.imshow("Window", resized_frame)
        cv2.imshow("Detected Yellow Lines", result_frame)
        #cv2.imshow("Yellow Mask", yellow_mask)



        current_time = time.time()
        dt = (current_time-lasttime)
        #if dt == 0:
            #dt = 1e-6
        lasttime = current_time
    

        #PID#############################
        
        base_speed = 50.0  #Algkiirus
        #PID 
        integral += error * dt
        derivative = (error - previous_error) / dt
        previous_error = error
        
        control_signal = Kp * error + Ki * integral + Kd * derivative

        #Vel out
        left_speed = base_speed + control_signal
        right_speed = base_speed - control_signal
        #print("controll signal: ", round(control_signal,2))

        print(f"left speed: {round(left_speed,2)}, right speed : {round(right_speed,2)}")

        # left_msg = Float32()
        # left_msg.data = left_speed
        # left_pub.publish(left_msg)

        # right_msg = Float32()   
        # right_msg.data = right_speed
        # right_pub.publish(right_msg)  
        

        

        # fps
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    # except KeyboardInterrupt:
        # print("Shutting down")
        # break

cap.release()
cv2.destroyAllWindows()


