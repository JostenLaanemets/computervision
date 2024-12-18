#!/usr/bin/env python3

from sensor_msgs.msg import NavSatFix
import math
import rospy
import csv
import pymap3d as pm
import pandas as pd
from std_msgs.msg import Float32

class Robot:
    def __init__(self):
        rospy.Subscriber('/gps/fix', NavSatFix, self.callback)

        #rataste publisher
        self.left_pub = rospy.Publisher('Left_vel', Float32, queue_size=10)
        self.right_pub = rospy.Publisher('Right_vel', Float32, queue_size=10)
        
    def callback(self, data):
        self.latitude = data.latitude
        self.longitude = data.longitude
        self.altitude = data.altitude

    def get_enu(self):
        self.enu = pm.geodetic2enu(self.latitude,self.longitude,self.altitude, 58.3428685594, 25.5692475361, 91.357)
        return self.enu
    
    def find_nearest_point(self, points):
            min_distance = None
            for point in points:
                # MSG TYPE PEAKS TWIST OLEMA MITTE FLOAT32
                distance = math.sqrt((point[0] - self.enu[0])**2 + (point[1] - self.enu[1])**2)
                if distance < min_distance:
                    min_distance = distance
                return point

    def send_vel(self, left, right):
        left_msg = Float32()
        left_msg.data = left
        self.left_pub.publish(left_msg)

        right_msg = Float32()
        right_msg.data = right
        self.right_pub.publish(right_msg)    

#Heading 
def calculate_heading(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    bearing = math.atan2(delta_y, delta_x)
    bearing = math.degrees(bearing)
    heading = (bearing + 360) % 360
    return heading

#Error angle
def calculate_error_angle(current_heading, target_bearing):
    error_angle = (target_bearing - current_heading + 360) % 360
    if error_angle > 180:
       error_angle -= 360
    return error_angle

lasttime = 0
i = 0
#PID 
Kp= 0.8
Ki = 0.01
Kd = 0.05

integral = 0.0
previous_error = 0.0
dt = 1e-6

points = [] 
with open('/home/roisten/catkin_ws/recordings/03-06-2024-14-21.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:                         
            x = float(row[0])                                        
    
        
            y = float(row[1])
            z = float(row[2])
            points.append([x, y, z])

robot = Robot()
rospy.init_node('Robot', anonymous=True)

while not rospy.is_shutdown():
    try:
        #print(robot.get_enu())
        prev = robot.get_enu()
        prevX, prevY, prevZ = prev
        
        current_time = rospy.get_time()
        dt = (current_time-lasttime)
        #if dt == 0:
            #dt = 1e-6
        lasttime = current_time
        x, y, z = robot.get_enu()

        #PID#############################
        if x != prevX or y != prevY:

            heading =calculate_heading(prevX, prevY, x, y)
            prevX , prevY = x , y
            marker_heading = calculate_heading(x, y ,points[i][0],points[i][1])
            error = calculate_error_angle(heading,marker_heading)
            
            base_speed = 50.0  #Algkiirus
            #PID 
            integral += error * dt
            derivative = (error - previous_error) / dt
            previous_error = error
            
            control_signal = Kp * error + Ki * integral + Kd * derivative


            #Vel out
            left_speed = base_speed + control_signal
            right_speed = base_speed - control_signal

            print(left_speed,right_speed)
            #publisher, mis loobib arduinole left ja right rataste väärtused!!!
            robot.send_vel(left_speed, right_speed)

            #############
            #Threshold
            Inx = points[i][0] - x 
            Iny = points[i][1] - y
            In = math.sqrt(Inx**2+Iny**2)
            print(In)
            if In<=10:
                    i= i+1

    except KeyboardInterrupt:
        print("Shutting down")
        break

