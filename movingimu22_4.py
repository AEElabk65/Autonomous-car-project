import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import serial
import time
import threading
import board
import adafruit_bno055
from picamera2 import Picamera2
ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
steering_angles = deque(maxlen=5)
steering_angles_1=[]
i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)
previous_yaw = None
low_speed_sent = False
normal_speed_sent = False
high_speed_sent = False
yaw_threshold=4
previous_time=time.time()
def load_image(image_path):
    # Load an image from the specified file
    return cv2.imread(image_path)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # Đảm bảo line là mảng 1 chiều có 4 phần tử
            if not isinstance(line, np.ndarray):
                line = np.array(line)
            if line.ndim == 2 and line.shape[0] == 1:
                line = line.flatten()
            
            # Kiểm tra xem line có đủ 4 phần tử không
            if len(line) == 4:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            else:
                print("Đoạn thẳng không hợp lệ:", line)

    return line_image


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * 1/2 # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * 1/2 # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))

            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        lane_lines.append(make_points(frame, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):       
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]
   
def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 7
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    # reduce the noise of the image: if it’s not between the 5-by-5 Kernel it is deleted
    edges = cv2.Canny(blur, 100, 200) #100 and 200 are min and max gradient intensities
    return edges

def region_of_interest(canny):
    height = canny.shape[0] #first dimension value
    width = canny.shape[1] #second dimension value
    mask = np.zeros_like(canny) #mask initialization
    shape = np.array([[(50, height), (width-50, height), (width-50, 300), (50,300)]], np.int32) #polygon 30
    cv2.fillPoly(mask, shape, 255) #mask with polygon size
    masked_image = cv2.bitwise_and(canny, mask) #final result
    # cv2.imshow("Canny mainlane", masked_image)
    cv2.waitKey(1)
    return masked_image

def detect_line_segments(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 20
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, np.array([]), minLineLength=60, maxLineGap=200)
    return line_segments

def get_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape
    if len(lane_lines) == 2:
        left_x1, left_y1, left_x2, left_y2 = lane_lines[0][0]
        right_x1, right_y1, right_x2, right_y2 = lane_lines[1][0]
        slope_l=math.atan((left_x1-left_x2) / (left_y1-left_y2))
        slope_r=math.atan((right_x1-right_x2) / (right_y1-right_y2))
        slope_ldeg = int(slope_l * 180.0 / math.pi)
        steering_angle_left = slope_ldeg  
        slope_rdeg = int(slope_r * 180.0 / math.pi)
        steering_angle_right = slope_rdeg
        if left_x2>right_x2: #horizontal line 
            if abs(steering_angle_left) <= abs(steering_angle_right):
                x_offset = left_x2 - left_x1
                y_offset = int(height / 2)
            elif abs(steering_angle_left) > abs(steering_angle_right):
                x_offset = right_x2 - right_x1
                y_offset = int(height / 2)
        else: #normal left line
                mid = int(width / 2)
                x_offset = (left_x2 + right_x2) / 2 - mid
                y_offset = int(height / 2)
    elif len(lane_lines) == 1:
                x1, _, x2, _ = lane_lines[0][0]
                x_offset = x2 - x1
                y_offset = int(height / 2)
    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)     
    #angle_to_mid_radian = math.atan(x_offset / y_offset)
    alfa = 0.6
    angle_to_mid_radian =(1-alfa)*math.atan(x_offset/ y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg +90
    angle = angle_to_mid_radian
    return steering_angle

def display_heading_line(frame, steering_angle, line_color=(0, 255,0), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 1.75)
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image


def send_command(msgID,steering_angles):
    """Send the first steering angle from the queue to Nucleo."""
    print(msgID,steering_angles)
        # Lấy giá trị đầu tiên từ hàng đợi mà không xóa nó
    content = steering_angles  # Sử dụng popleft() để lấy và xóa.
    command = f"#{msgID}:{content};;\r\n"  # Structure of command from RPi
    ser.write(command.encode())
    time.sleep(0.5)  # Delay for Nucleo to response
    response = ser.readline().decode().strip()
    print(f"Response from Nucleo: {response}")
    # Xóa giá trị đã gửi khỏi hàng đợi
    
def check_yaw_change(current_yaw,current_pitch):
    global previous_yaw,yaw_threshold,low_speed_sent,normal_speed_sent, high_speed_sent
    if previous_yaw is None:
        previous_yaw = current_yaw
        return
    yaw_change = abs(current_yaw - previous_yaw)
    if yaw_change > yaw_threshold and current_pitch>-6 and current_pitch<6:
        if not low_speed_sent:
            send_command(1, 10)
            low_speed_sent = True
            normal_speed_sent = False
            high_speed_sent = False
            
    if yaw_change < yaw_threshold and current_pitch>-6 and current_pitch<6:
        if low_speed_sent:
            send_command(1, 20)
            low_speed_sent = False
            normal_speed_sent = False
            high_speed_sent = False
    if yaw_change < yaw_threshold and pitch > 6 and not high_speed_sent :
        send_command(1, 6)
        high_speed_sent = True
        normal_speed_sent = False
        low_speed_sent = True
#     elif -6 < pitch < 8 and not normal_speed_sent:
#         send_command(1, 15)
#         normal_speed_sent = True
#         high_speed_sent = False
#         low_speed_sent = False
    if yaw_change < yaw_threshold and pitch < -6 and not low_speed_sent:
        send_command(1, 6)
        low_speed_sent = True
        normal_speed_sent = False
        high_speed_sent = False
        temp=0
            
    previous_yaw = current_yaw
    return yaw_change
# def handle_pitch(pitch):
#     global previous_yaw,yaw_threshold,low_speed_sent,normal_speed_sent, high_speed_sent
#     if pitch > 5 and not high_speed_sent and :
#         send_command(1, 8)
#         high_speed_sent = True
#         normal_speed_sent = False
#         low_speed_sent = False
#         temp=0# def handle_pitch(pitch):
#     global previous_yaw,yaw_threshold,low_speed_sent,normal_speed_sent, high_speed_sent
#     if pitch > 5 and not high_speed_sent and :
#         send_command(1, 8)
#         high_speed_sent = True
#         normal_speed_sent = False
#         low_speed_sent = False
#         temp=0
# #     elif -6 < pitch < 8 and not normal_speed_sent:
# #         send_command(1, 15)
# #         normal_speed_sent = True
# #         high_speed_sent = False
# #         low_speed_sent = False
#     elif pitch < -6 and not low_speed_sent:
#         send_command(1, 6)
#         low_speed_sent = True
#         normal_speed_sent = False
#         high_speed_sent = False
#         temp=0
# #     elif -6 < pitch < 8 and not normal_speed_sent:
# #         send_command(1, 15)
# #         normal_speed_sent = True
# #         high_speed_sent = False
# #         low_speed_sent = False
#     elif pitch < -6 and not low_speed_sent:
#         send_command(1, 6)
#         low_speed_sent = True
#         normal_speed_sent = False
#         high_speed_sent = False
#         temp=0

def send_commands_from_queue():
    while True:
        if steering_angles:
            if steering_angles[0]>25:
                send_command(2, 25)
            if steering_angles[0]<-25 :
                send_command(2, -25)
            else:
                send_command(2, steering_angles[0])  # Gửi góc lái đầu tiên trong hàng đợi  # Loại bỏ phần tử đã gửi khỏi hàng đợi

                time.sleep(0.01)  # Đợi 0.5 giây trước khi gửi lệnh tiếp theo
        else:
            time.sleep(0.5)  # Nếu không có phần tử nào trong hàng đợi, chờ một chút trước khi kiểm tra lại

# Khởi tạo và bắt đầu thread
command_thread = threading.Thread(target=send_commands_from_queue)
command_thread.daemon = True  # Đặt là daemon thread để chương trình có thể thoát khi chỉ còn daemon threads
command_thread.start()

cv2.startWindowThread()

picam2 = Picamera2()
#picam2 = cv2.VideoCapture(0)
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
height = 704
width = 1279
send_command(1,25)
old_lines = None 
while True:
    lane_line = picam2.capture_array()
    #lane_line = picam2.read() 
    frame =cv2.resize(np.copy(lane_line) , (width , height))
    canny_image = detect_edges(frame)
    cropped_canny = region_of_interest(canny_image)
    lines  = detect_line_segments(cropped_canny)
    
    try:
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        steering_angle = get_steering_angle(frame, averaged_lines)
        heading_image = display_heading_line(line_image,steering_angle)
        combo_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
        steering_angle = steering_angle - 90
        steering_angles.append(int(steering_angle * 0.8))
        steering_angles_1.append(steering_angle)
        old_lines = averaged_lines
#         cv2.imshow('result', combo_image)
        x,y,z = sensor.acceleration
        current_time =time.time()-previous_time
        if current_time >=1:   
            yaw, pitch, _ = sensor.euler
            previous_time = time.time()
            yaw_change = check_yaw_change(yaw,pitch)
#         if yaw_change is not None:	
#           if yaw_change<yaw_threshold:
#              handle_pitch(pitch)
             
    except:
        if old_lines is not None:
            line_image = display_lines(frame, old_lines)
            combo_image = cv2.addWeighted(frame, 0.3, heading_image, 1, 1)
            steering_angle = 90;
    #    cv2.imshow('result', combo_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
#         txt_filename = 'steering_angles2.txt'
#         with open(txt_filename, 'w') as file:
#             for angle in steering_angles_1:
#                 file.write(str(angle) + '\n')
#         print(f"All steering angles saved to '{txt_filename}'")
        break
send_commands_from_queue() # Sử dụng ID 2 cho góc lái 
time.sleep(0.5)

# with open(txt_filename, 'w') as file:
#     for angle in steering_angles:
#         file.write(str(angle) + '\n')
# print(f"All steering angles saved to '{txt_filename}'")




