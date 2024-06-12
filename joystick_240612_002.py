import os
import pygame
import cv2
import logging
import datetime
from typing import Sequence
from jetracer.nvidia_racecar import NvidiaRacecar
import sys
from pathlib import Path
import torch
import torchvision
from torchvision import models
from PIL import Image
from cnn.center_dataset import TEST_TRANSFORMS
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np
import Jetson.GPIO as GPIO

########### 카메라 설정 ##################

#전방카메라
sys.path.append('/home/ircv6/HYU-2024-Embedded/week07/Camera')
import camera
cam = camera.Camera(
        sensor_id = 0,
        window_title = 'Camera',
        save_path = 'record',
        save = True,
        stream = False,
        log = False)

# 후방카메라
cam2 = cv2.VideoCapture(1)
cam2.set(cv2.CAP_PROP_FPS,30)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

########## 차량 연결 ################
car = NvidiaRacecar()

########## GPIO 핀 번호 설정 ##########
GPIO.cleanup()

GPIO.setmode(GPIO.BCM)
trig1, echo1 = 26, 20
trig2, echo2 = 19, 16
trig3, echo3 = 6, 13
trig4, echo4 = 5, 12
trig5, echo5 = 17,27

GPIO.setup(trig1, GPIO.OUT)
GPIO.setup(echo1, GPIO.IN)
GPIO.setup(trig2, GPIO.OUT)
GPIO.setup(echo2, GPIO.IN)
GPIO.setup(trig3, GPIO.OUT)
GPIO.setup(echo3, GPIO.IN)
GPIO.setup(trig4, GPIO.OUT)
GPIO.setup(echo4, GPIO.IN)
GPIO.setup(trig5, GPIO.OUT)
GPIO.setup(echo5, GPIO.IN)

RR_origin = 50
LR_origin = 50
RF = 50
R1 = 50
R2 = 50

timeout = 0.05831
timesleep = 0.0005
def measure_distance(trig, echo, prev_distance, timeout):
    
    sensitivity=0.6

    GPIO.output(trig, False)
    time.sleep(0.00001)
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    start_time = time.time()
    pulse_start, pulse_end = time.time(), time.time()

    while GPIO.input(echo) == 0:
        pulse_start = time.time()
        if pulse_start - start_time > timeout:
            return 1000  # 초과 시간을 설정하여 최대값을 반환

    while GPIO.input(echo) == 1:
        pulse_end = time.time()
        if pulse_end - start_time > timeout:
            return 1000  # 초과 시간을 설정하여 최대값을 반환

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    distance = prev_distance * (1-sensitivity) + distance * (sensitivity)
    return distance


# def detect_harris_corners(image):
#     # 그레이스케일로 변환
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # float32로 변환
#     gray = np.float32(gray)

#     # 해리스 코너 검출기 적용
#     corners = cv2.cornerHarris(gray, 2, 3, 0.04)

#     # 결과를 dilate해서 코너 표시를 더 잘 보이게 함
#     corners = cv2.dilate(corners, None)

#     # 코너 검출 결과를 원본 이미지에 표시
#     image[corners > 0.01 * corners.max()] = [0, 0, 255]

#     return image

# def detect_edges_and_lines(image):
#     # 그레이스케일로 변환
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # 가우시안 블러 적용 (노이즈 제거)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Canny 에지 검출기 적용
#     edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

#     # Hough 변환을 사용하여 선 검출
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

#     # 검출된 선을 원본 이미지에 그리기
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             # 수직선 필터링
#             if abs(x1 - x2) < 10:  # 수직선 기준 설정 (10은 임계값)
#                 cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

#     return image

def detect_line(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ###### 파란색의 범위
    # lower_blue = np.array([100, 150, 0])
    # upper_blue = np.array([140, 255, 255])
    # ###### 청회색의 HSV 범위
    # lower_cyan_gray = np.array([100, 50, 100])
    # upper_cyan_gray = np.array([120, 80, 140])
    
    # 파란색 & 청회색 HSV
    # base_hsv = np.array([110, 63, 119])
    # lower_hsv = base_hsv - np.array([10, 20, 30])
    # upper_hsv = base_hsv + np.array([10, 20, 30])
    # lower_hsv = np.clip(lower_hsv, 0, 255)
    # upper_hsv = np.clip(upper_hsv, 0, 255)

    
    # mask = cv2.inRange(img, lower_blue, upper_blue)
    # mask2 = cv2.inRange(img, lower_cyan_gray, upper_cyan_gray)
    # combined_mask = cv2.bitwise_or(mask, mask2)

    # mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    # 빨간색 HSV
    # 첫 번째 빨간색 HSV 범위
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # 두 번째 빨간색 HSV 범위
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 두 범위의 마스크 생성
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

    # 두 마스크 결합
    combined_mask_red = cv2.bitwise_or(mask1, mask2)

    result = cv2.bitwise_and(img, img, mask=combined_mask_red)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, bin_result = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_points = []
    for contour in contours:
        for point in contour:
            all_points.append(point[0])
    if len(all_points) == 0:
        return 0, 0, 0, 0

    all_points = np.array(all_points)
    x, y, w, h = cv2.boundingRect(all_points)
    
    return x, y, w, h



########## 기계학습 모델 로딩 #############
device = torch.device('cuda')
model_straight = models.alexnet(num_classes=2, dropout=0.0)
model_straight.load_state_dict(torch.load('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_straight_370_v4.pth')) #('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_left_370.pth')) #('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_straight_320_v1.pth'))
model_straight = model_straight.to(device)

# model_left = models.alexnet(num_classes = 2, dropout = 0.0)
# model_left.load_state_dict(torch.load('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_left_370_v4.pth')) #('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_left_v1.pth'))
# model_left = model_left.to(device)

# model_right = models.alexnet(num_classes = 2, dropout = 0.0)
# model_right.load_state_dict(torch.load('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_right_370_v4.pth')) #('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_right_320_v1.pth'))
# model_right = model_right.to(device)

#처음 시작 시, 직진 모델 적용
model = model_straight

model_yolo = YOLO("240611_001.pt", task='detect',verbose = False).to(device)
classes = model_yolo.names

########### 저장 경로 설정 #################
save_path = str('record')
# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

car.throttle_gain = 0.37
car.steering_gain = -1

joystick = pygame.joystick.Joystick(0)
joystick.init()

running = True
throttle_range = (-0.5, 0.5)

############# 플래그 설정 ##############

# 조이스틱 버튼 플래그
flag_4_1, flag_4_2=False, False
flag_0_1, flag_0_2=False, False

# 1번 : 카메라 녹화 버튼
flag_1_1, flag_1_2=False, False

# 3번 : 게인 모드에서 조절 버튼
flag_3_1,flag_3_2=False, False

# 6번 : 자율 주행
flag_6_1, flag_6_2 =False, False

#7번 : Throttle / 제어 Gain 설정 변경 
flag_7_1, flag_7_2=False, False

automode = False
collecting = False
cnt = 0
gain_Tuning = False
gain_number = 0
gain_name = "Kp"

## 제어 파라미터
th_control=0.4
throttle_txt=th_control
steering_input=0
deadzone=10
K_p=0.0018
K_i=1e-6
K_d=0.0038
K_a=0
error_I=0
error_prev=0

## 스티어링 제어 함수
def steering_control(reference,K_p,K_i,K_d,K_a):
    global error_I, error_prev
        
    error_feedback = reference
    error_P = error_feedback
    error_I += error_feedback
    error_D = error_feedback-error_prev
    input_pid = K_p*error_P + K_i*error_I +K_d*error_D

    input_windup = max(-1,min(0.9,input_pid))
    error_windup = input_pid - input_windup
    error_I -= K_a * error_windup
    error_prev=error_feedback
    return input_windup

x, y, sampling, box_size_criterion, box_size_max_criterion = 0,0,0,5000, 20000
bus, bus_cnt, n_bus_cnt, bus_time, n_bus_time = False, 0, 5000, 70, 200
cross, cross_cnt, n_cross_cnt, cross_time, n_cross_time = False, 0, 5000, 50, 200
left, right, straight, model_num = False, False, False, 0


def draw_boxes_sign(image, label, x1,y1,x2,y2):
    # x1, y1, x2, y2 = map(int, box.xyxy[0])
    # score = round(float(box.conf[0]), 2)
    # label = int(box.cls[0])
    classes = ['bus','crosswalk', 'left', 'right', 'straight']
    cls_name = classes[label]
    color = [(0,0,0),(255,255,255), (255,0,0), (0,255,0), (0,0,255)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color[label], 2)
    cv2.putText(image, f"{cls_name}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[label], 1, cv2.LINE_AA)
reference = 0
label,x1,y1,x2,y2 = 0,0,0,0,0

def draw_boxes(image, x1, y1, x2, y2):
    cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)


def draw_center(image, cx, cy):
    ####cv2.circle(image, center, radius, color, thickness=None, lineType=None, shift=None)
    width, height, _ = image.shape
    dx, dy = width/2-cx, height/2-cy
    cv2.circle(image,(cx, cy), 1, (0, 255, 0), 5)
    if dx < 0:
        cv2.putText(image, f"dx: {dx}.", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(image, f"dx: {dx}.", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

def change_direction(curr_sign):
    if curr_sign>=0:
        gain=-1
    else:
        gain=1

    for i in range(0,38):
        car.throttle = gain * i * 0.01
        time.sleep(0.01)
        # print(f'throttle: {car.throttle}, steering: {car.steering}, state: {state}')
    car.throttle=0
    time.sleep(0.5)

# label,x1,y1,x2,y2 = 0,0,0,0,0

# 주차장 중심 인식 좌표 
cx, cy, score = 0, 0, 0
x, y, w, h = 0, 0, 0, 0
# label, x1, y1, x2, y2 = 0, 0, 0, 0, 0
# 전체 차량의 상태를 제어하는 상태변수
state=0 # 0: 정상 주행, 1: 주차공간 탐지를 실행 2: 정지 후, 회전, 3: 후진하며 회전, 4: 조향을 틀고, 후진 5: 주차 완료 6: 재정렬 7: 비상정지
parking_flag=False
center = False
test_count=0 
counter_2_1 = 0
reference = 0
w,h=0,0 # 상자 너비, 길이 초기화
camera2 = False

while running:
    start_time = time.time()
    pygame.event.pump()
    ################ 자율 주행 모드 ###############
    if automode:
        # print(n_bus_cnt, n_cross_cnt)
        ret, frame = cam.cap[0].read()
        if ret:
            pil_image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            width, height = pil_image.size
            
            ########## 초음파 센서 거리 측정 #############
            if state>=1:
                RR_origin = measure_distance(trig1, echo1,RR_origin, timeout)
                #time.sleep(timesleep)
                RF = measure_distance(trig2, echo2,RF, timeout)
                #time.sleep(timesleep)
                R2 = measure_distance(trig3, echo3,R2, timeout)
                #time.sleep(timesleep)
                LR_origin = measure_distance(trig4, echo4,LR_origin, timeout)
                #time.sleep(timesleep)
                R1 = measure_distance(trig5, echo5, R1, timeout)
                #time.sleep(timesleep)

                ## 오프셋 처리
                RR = max(RR_origin-5,0)
                LR = max(LR_origin-4.5,0)

                print(f'Right front : {RF} cm, Right Rear: {RR} cm, Left Rear: {LR} cm,Rear 1(left): {R1} Rear 2(right): {R2} cm, state : {state}')
                # print(f'throttle: {car.throttle}, steering: {car.steering}, state: {state}')
            
            if state>=2:
                ret2, frame2 = cam2.read()
                frame2 = cv2.resize(frame2, (width, height))
                if ret2:
                    pil_image2 = Image.fromarray(cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB))
                    # pil_image2 = Image.fromarray(cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB))
            
            if state==0 or state==1:
                with torch.no_grad():
                    img = TEST_TRANSFORMS(pil_image).to(device)
                    img = img[None, ...]
                    output = model(img).detach().cpu().numpy()
                    x, y = output[0]
                    x = (x / 2 + 0.5) * width
                    y = (y / 2 + 0.5) * height
                    # print(model,"'s output :",x,',',y)    
                    # print('model',model_num,'is running')

                    ## 조향 제어 로직(pid)
                    reference = x - 475
                    steering_input = steering_control(reference,K_p,K_i,K_d,K_a)
                    car.steering = steering_input 
                
                    ## Throttle Control
                    #버스 플래그
                    if bus == True and bus_cnt < bus_time:
                        if abs(steering_input)>=0.7:
                            throttle_txt= th_control-0.028
                            car.throttle = th_control-0.028
                        else:
                            throttle_txt = th_control - 0.031
                            car.throttle = th_control-0.031
                    #횡단보도 플래그
                    elif cross == True and cross_cnt < cross_time:
                        car.throttle = 0
                    #정상 주행
                    elif bus == False and cross == False:
                        if abs(steering_input)>=0.65:
                            throttle_txt = th_control+0.01
                            car.throttle = th_control+0.01
                        else:
                            throttle_txt = th_control
                            car.throttle = th_control
                        # elif abs(steering_input)<=0.4 and n_cross_cnt < n_cross_time:
                        #     throttle_txt = th_control-0.01
                        #     car.throttle = th_control-0.01
                        # else:
                        #     throttle_txt = th_control
                        #     car.throttle = th_control
                    
                    ######### YOLOv8 ##########
                    if sampling==9:
                        sampling=0
                        pred = model_yolo(pil_image, stream = False, device = device)
                        class_list = []
                        box_size_list = []
                        for r in pred:
                            if r.boxes:
                                for box in r.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    # score = round(float(box.conf[0]), 2)
                                    label = int(box.cls[0])
                                    # color = colors[label].tolist()
                                    cls_name = classes[label]
                                    class_list.append(cls_name)

                        box_size = abs(x2-x1)*abs(y2-y1)

                        # 버스 표지판을 인지하고, 안본지 오래되었을 때 잠깐 서행
                        if 'bus' in class_list and bus == False and n_bus_cnt >= n_bus_time and box_size >= box_size_criterion:
                            bus = True
                            n_bus_cnt = 0
                            
                        # 잠깐 정지
                        elif 'crosswalk' in class_list and cross == False and n_cross_cnt >= n_cross_time and box_size >= box_size_criterion:
                            cross = True
                            n_cross_cnt = 0

                        # 모델 변경
                        if 'left' in class_list and box_size >= box_size_criterion and box_size < box_size_max_criterion:
                            model = model_left
                            model_num = 1
                        if 'right' in class_list and box_size >= box_size_criterion and box_size < box_size_max_criterion:
                            model = model_right
                            model_num = 2
                        if 'straight' in class_list and box_size >= box_size_criterion and box_size < box_size_max_criterion:
                            model = model_straight
                            model_num = 0
                    else:
                        sampling += 1

                        ####### 버스 ##########
                        if bus == True and bus_cnt < bus_time:
                            bus_cnt += 1
                            print("bus detected",'bus_cnt :',bus_cnt, 'n_bus_cnt:',n_bus_cnt)
                        
                        elif bus_cnt >= bus_time and bus == True:
                            bus = False
                            bus_cnt = 0
                            n_bus_cnt +=1

                        if bus == False and n_bus_cnt < n_bus_time:
                            n_bus_cnt += 1
                        
                        ####### 횡단보도 #########
                        if cross == True and cross_cnt < cross_time:
                            cross_cnt += 1
                            print("crosswalk detected",'crosswalk_cnt :',cross_cnt, 'n_crosswalk_cnt:',n_cross_cnt)

                        elif cross_cnt >= cross_time and cross == True:
                            cross = False
                            cross_cnt = 0
                            n_cross_cnt += 1

                        if cross == False and n_cross_cnt < n_cross_time:
                            n_cross_cnt += 1

                        # ######### 좌회전 #########
                        # if left == True:
                        #     model = model_left
                        #     model_num = 1
                        # ######### 우회전 ##########
                        # if right == True:
                        #     model = model_right
                        #     model_num = 2
                        # ######### 직진 ##########
                        # if straight == True:
                        #     model = model_straight
                        #     model_num = 0
                    # 모니터링
                    if gain_number==0:
                        gain_name="Kp"
                    elif gain_number==1:
                        gain_name="Ki"
                    elif gain_number==2:
                        gain_name="Kd"
                    else:
                        gain_name="Ka"

                    if state==0:
                        if parking_flag and abs(steering_input)<0.1:
                            state=1
                    ########## 표지판을 인지했을 때 state=1 ########
                    elif state==1:
                        
                        ######## 초음파 센서로 주차 공간 인식 시 #########
                        if 35<RF<50 and RR<20:
                            state=2.1
                            car.throttle=0
                            car.steering= -1
                            time.sleep(2)
            ####### [2.1] : 1) 3초동안 앞으로 좌회전           
            # elif state==2.1:
            #     car.steering= -1
            #     car.throttle=th_control+0.05
            #     time.sleep(1.2)
                
            #     car.steering=1
            #     car.throttle=0
            #     time.sleep(1)
            #     # print(f'throttle: {car.throttle}, steering: {car.steering}, state: {state}')
            #     state=2.15

            ####### [2.1] : 2) center 스티커 인지할 때 까지 좌회전
            elif state == 2.1:

                car.steering = -1
                car.throttle = th_control #+ 0.01

                ###### detect_line 함수 결과값으로, 사각형 x, y, w, h 출력
                ###### 검출 안되면 0으로 출력
                x, y, w, h = detect_line(frame2)

                if w*h >= 100:
                    center=True
                    
                if center:
                    car.throttle = 0
                    time.sleep(1.2)
                    car.steering = 1
                    time.sleep(1)
                    change_direction(1) # 전진->후진
                    state = 2.2

                # # while center == False:
                # #     car.throttle = th_control + 0.05
                # if center:
                #     car.throttle = 0
                #     time.sleep(1.2)
                #     car.steering = 1
                #     time.sleep(1)
                #     change_direction(1) # 전진->후진
                #     state = 2.2

            # elif state==2.15:
            #     car.steering=1
            #     for i in range(0,38):
            #         car.throttle = 0 - i * 0.01
            #         time.sleep(0.01)
            #         # print(f'throttle: {car.throttle}, steering: {car.steering}, state: {state}')
            #     car.throttle=0
            #     time.sleep(0.5)
            #     state=2.2
            ####### 후진, 우회전
            elif state==2.2:
                car.steering=1
                car.throttle=-0.46
                if LR+RR < 25:
                    counter_2_1+=1
                    if counter_2_1<5:
                        state=2.3
                        car.steering=0
                        car.throttle=0
                        time.sleep(1)
                else:
                    counter_2_1=0

            ####### 핸들 정렬 후 후진
            elif state==2.3:
                car.throttle=-0.46
                if abs(R1-R2)<5:
                    car.steering=0
                elif LR-RR>2:
                    car.steering=-0.5
                elif LR-RR<-2:
                    car.steering=0.5
                else:
                    car.steering=0
                
                if R1< 3.5 or R2<3.5:
                    state = 3
                #time.sleep(2)
                # print(f'throttle: {car.throttle}, steering: {car.steering}, state: {state}')

            elif state ==3:
                car.throttle = 0
                car.steering = 0
                w,h
                automode=False
                time.sleep(2)
                    
    ############### 조이스틱 조작 모드 ###############
    else:
        error_I=0 # pid 제어에서 사용한 에러 누적값 초기화
        error_prev=0 
        state=0 #주차 상태 0으로 초기화
        parking_flag=0 #parking 상태 0으로 초기화
        throttle = -joystick.get_axis(1)
        throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        car.throttle = throttle
        steering = joystick.get_axis(2)
        car.steering = steering
        # print("thro: {:2f} steer input: {} ".format(throttle, steering))
    
    #print("thro: {:2f} steer: {:2f} th_gain: {:2f} deadzone: {} ".format(throttle, car.steering_gain*steering,car.throttle_gain,deadzone))
    
    
    if joystick.get_button(11): # start button
        running = False

    flag_7_1 = joystick.get_button(7)
    if flag_7_1 and not flag_7_2:
        gain_Tuning = not gain_Tuning
        flag_4_1=flag_4_2=False #4번 버튼 초기화    
    flag_7_2=flag_7_1 

    if gain_Tuning: # 튜닝 모드 조작
        # 튜닝할 게인 번호 조정: 0은 K_p, 1은 K_i, 2는 K_d, 3은 K_a
        # 게인 번호 증가
        flag_1_1 = joystick.get_button(1)
        if flag_1_1 and not flag_1_2:
            gain_number= (gain_number+1)%4
        flag_1_2=flag_1_1

        flag_3_1 = joystick.get_button(3)
        if flag_3_1 and not flag_3_2:
            gain_number= (gain_number-1)%4
        flag_3_2=flag_3_1

        if gain_number==0:
            # K_p 증가
            flag_4_1 = joystick.get_button(4)
            if flag_4_1 and not flag_4_2:
                K_p += 0.001
                K_p = min(10,K_p)
            flag_4_2=flag_4_1   

            # K_p 감소
            flag_0_1 = joystick.get_button(0)
            if flag_0_1 and not flag_0_2:
                K_p -= 0.001
                K_p = max(0,K_p)
            flag_0_2=flag_0_1

        elif gain_number==1:
            # K_i 증가
            flag_4_1 = joystick.get_button(4)
            if flag_4_1 and not flag_4_2:
                K_i += 0.0001
                K_i = min(10,K_i)
            flag_4_2=flag_4_1   

            # K_i 감소
            flag_0_1 = joystick.get_button(0)
            if flag_0_1 and not flag_0_2:
                K_i -= 0.0001
                K_i = max(0,K_i)
            flag_0_2=flag_0_1

        elif gain_number==2:
            # K_d 증가
            flag_4_1 = joystick.get_button(4)
            if flag_4_1 and not flag_4_2:
                K_d += 0.001
                K_d = min(10,K_d)
            flag_4_2=flag_4_1   

            # K_d 감소
            flag_0_1 = joystick.get_button(0)
            if flag_0_1 and not flag_0_2:
                K_d -= 0.001
                K_d = max(0,K_d)
            flag_0_2=flag_0_1
        else: # gain_number==3
            # K_a 증가
            flag_4_1 = joystick.get_button(4)
            if flag_4_1 and not flag_4_2:
                K_a += 0.001
                K_a = min(10,K_a)
            flag_4_2=flag_4_1   

            # K_a 감소
            flag_0_1 = joystick.get_button(0)
            if flag_0_1 and not flag_0_2:
                K_a -= 0.001
                K_a = max(0,K_a)
            flag_0_2=flag_0_1

    else: #드라이빙 모드 조작
        #녹화버튼 설정
        if joystick.get_button(1) and not flag_1_1:
            collecting = not collecting
            flag_1_1 = True
            if collecting:
                print("Starting image Collection...")
            else:
                print("Stopping image Collection...")
                
        elif not joystick.get_button(1) and flag_1_1:
            flag_1_1 = False

        # 자율주행 모드 시 throttle 증가
        flag_4_1 = joystick.get_button(4)
        if flag_4_1 and not flag_4_2:
            th_control += 0.01
            th_control = min(0.7,th_control)
        flag_4_2=flag_4_1   

        # 자율주행 모드 시 throttle 감소
        flag_0_1 = joystick.get_button(0)
        if flag_0_1 and not flag_0_2:
            th_control -= 0.01
            th_control = max(0,th_control)
        flag_0_2=flag_0_1

        # state=0 초기화
        flag_3_1 = joystick.get_button(3)
        if flag_3_1 and not flag_3_2:
            parking_flag= not parking_flag
        flag_3_2=flag_3_1

    end_time = time.time() 
    compute_time = end_time - start_time
    #print(compute_time*1000)
    box_size = abs(x2-x1)*abs(y2-y1)
    # print('box size :', box_size)
    # 사진 수집 버튼 인식
    cam_save_path = Path(cam.save_path)
    parent_folder = cam_save_path.name
    if collecting:
        ret, frame = cam.cap[0].read()

        if camera2 == True:
            ret2, frame2 = cam2.read()

        if ret:
            # 전방 카메라 프레임 크기 얻기
            front_height, front_width, _ = frame.shape
        
            # 이미지 이름 생성
            timestamp = datetime.datetime.now().strftime('%Y%m%d')
            frame_path = cam.save_path / f"{parent_folder}_{cnt}.jpg"
            cnt += 1
            
            ######## 전방 카메라 #######
            #중앙선 예측 위치 점 찍기
            cv2.circle(frame, (int(x),int(y)),10,(255,0,0),-1)
            cv2.putText(frame, f'reference: {reference}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ####### 후방 카메라 #######
            # 후방 카메라 프레임 크기 조정
            # x, y, w, h = detect_line(frame2)
            # cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(frame2, f'blue range size :{w*h}', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )

            # 두 이미지를 수평으로 결합
            # combined_frame = np.hstack((frame, frame2))
            # cv2.imwrite(str(frame_path), combined_frame)  # 이미지 저장
            # cv2.imwrite(str(frame_path), frame2)
            cv2.imwrite(str(frame_path), frame)
            # print(f"Saved frame at {timestamp}")
            
        else:
            print("Failed to capture frame")
    
    # 자율주행 버튼 인식
    if joystick.get_button(6) and not flag_6_1:
        automode = not automode
        flag_6_1 = True
        if automode:
            print("AutoDriving Start...")
        else:
            print("AudoDriving Stopped...")
    elif not joystick.get_button(6) and flag_6_1:
        flag_6_1 = False
    
    

    #if not gain_Tuning:
        #print(f"Loop duration: {loop_duration:.4f} seconds, steering input: {steering_input:.4f}, x: {x},y: {y}")
    
camera.cap[0].release()
cv2.destroyAllWindows()
GPIO.cleanup()
pygame.quit()       
