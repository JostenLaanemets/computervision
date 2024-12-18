
import cv2
import numpy as np


cap = cv2.VideoCapture("/home/josten/Videos/Line.webm") 
success, image = cap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)



while success:
    success, image = cap.read()
    frame = cv2.resize(image, (640,480))
        
    tl = (220, 300)
    tr = (440, 300)
    bl = (110, 472)
    br = (550, 472)

    cv2.circle(frame, tl, 5, (0,0,255),-1)
    cv2.circle(frame, bl, 5, (0,0,255),-1)
    cv2.circle(frame, tr, 5, (0,0,255),-1)
    cv2.circle(frame, br, 5, (0,0,255),-1)

    pts1 = np.float32([tl,bl,tr,br])
    pts2 = np.float32([[0,0], [0, 480], [640, 0], [640, 480]])

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_frame = cv2.warpPerspective(frame,matrix, (640,480))

    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower  = np.array([l_h, l_s, l_v])
    higher = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv_transformed_frame, lower, higher)


    cv2.imshow('Line', frame)
    cv2.imshow('Birdseye', transformed_frame)
    cv2.imshow("skibidi", mask)
 

    # Exit on 'q' key press
    if cv2.waitKey(29) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
