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

########### 카메라 설정 ##################
sys.path.append('/home/ircv6/HYU-2024-Embedded/week07/Camera')
import camera
cam = camera.Camera(
        sensor_id = 0,
        window_title = 'Camera',
        save_path = 'record',
        save = True,
        stream = False,
        log = False)

########## 차량 연결 ################
car = NvidiaRacecar()

########## 기계학습 모델 로딩 부 #############
device = torch.device('cuda')
model_straight = models.alexnet(num_classes=2, dropout=0.0)
model_straight.load_state_dict(torch.load('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_straight_320_v1.pth'))
model_straight = model_straight.to(device)

model_left = models.alexnet(num_classes = 2, dropout = 0.0)
model_left.load_state_dict(torch.load('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_left_v1.pth'))
model_left = model_left.to(device)

model_right = models.alexnet(num_classes = 2, dropout = 0.0)
model_right.load_state_dict(torch.load('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_right_320_v1.pth'))
model_right = model_right.to(device)

#처음 시작 시, 직진 모델 적용
model = model_straight

model_yolo = YOLO("240515_002.pt", task='detect').to(device)
classes = model_yolo.names

########### 저장 경로 설정 #################
save_path = str('record')
# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

car.throttle_gain = 0.4
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
th_control=0.37
steering_input=0
deadzone=10
K_p=0.003
K_i=0
K_d=0.001
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

    input_windup = max(-1,min(1,input_pid))
    error_windup = input_pid - input_windup
    error_I -= K_a * error_windup
    error_prev=error_feedback
    return input_windup

#previous_frame=None
x, y = 0, 0
sampling=0

bus, bus_cnt, n_bus_cnt, bus_time, n_bus_time = False, 0, 5000, 50, 200
cross, cross_cnt, n_cross_cnt, cross_time, n_cross_time = False, 0, 5000, 50, 200
left, right, straight, model_num = False, False, False, 0

def draw_boxes(image, label, x1,y1,x2,y2):
    # x1, y1, x2, y2 = map(int, box.xyxy[0])
    # score = round(float(box.conf[0]), 2)
    # label = int(box.cls[0])
    classes = ['bus','crosswalk', 'left', 'right', 'straight']
    cls_name = classes[label]
    color = [(0,0,0),(255,255,255), (255,0,0), (0,255,0), (0,0,255)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color[label], 2)
    cv2.putText(image, f"{cls_name}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[label], 1, cv2.LINE_AA)

label,x1,y1,x2,y2 = 0,0,0,0,0
while running:
    # start_time = time.time()
    pygame.event.pump()
    ################ 자율 주행 모드 ###############
    if automode:
        # print(n_bus_cnt, n_cross_cnt)
        ret, frame = cam.cap[0].read()
        if ret:
            pil_image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            width, height = pil_image.size

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
                reference = x - 480
                steering_input = steering_control(reference,K_p,K_i,K_d,K_a)
                car.steering = steering_input 

                # #print("Steering_change")
                # if gain_Tuning: #튜닝 모드로 작동
                #     car.throttle = 0
                #     print("Tuning mode, gain_num: {} K_p: {}, K_i: {} K_d: {} K_a: {} x_error: {} y_axis: {} ".format(gain_name, K_p, K_i,K_d,K_a,reference,y))
                # else: # 드라이빙 모드로 작동
                #     car.throttle = th_control
                #     #print("Driving mode, thro: {:2f} steer input: {} K_p: {} x_error: {} y_axis: {} ".format(th_control, steering_input,K_p,reference,y))
                

                #버스 플래그
                if bus == True and bus_cnt < bus_time:
                    car.throttle = th_control - 0.05
                #횡단보도 플래그
                elif cross == True and cross_cnt < cross_time:
                    car.throttle = 0
                #정상 주행
                elif bus == False and cross == False:
                    car.throttle = th_control
                
                if sampling==10:
                    sampling=0
                    pred = model_yolo(pil_image, stream = False, device = device)
                    class_list = []
                    for r in pred:
                        if r.boxes:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                # score = round(float(box.conf[0]), 2)
                                label = int(box.cls[0])
                                # color = colors[label].tolist()
                                cls_name = classes[label]
                                class_list.append(cls_name)


                    # 버스 표지판을 인지하고, 안본지 오래되었을 때 잠깐 서행
                    if 'bus' in class_list and bus == False and n_bus_cnt >= n_bus_time:
                        bus = True
                        n_bus_cnt = 0

                    # 잠깐 정지
                    elif 'crosswalk' in class_list and cross == False and n_cross_cnt >= n_cross_time:
                        cross = True
                        n_cross_cnt = 0

                    # 모델 변경
                    if 'left' in class_list:
                        # left, right, straight = True, False, False
                        model = model_left
                        model_num = 1
                    if 'right' in class_list:
                        # left, right, straight = False, True, False
                        model = model_right
                        model_num = 2
                    if 'straight' in class_list:
                        # left, right, straight = False, False, True
                        model = model_straight
                        model_num = 0
                else:
                    sampling += 1

                    ####### 버스 ##########
                    if bus == True and bus_cnt < bus_time:
                        bus_cnt += 1
                        # print("bus detected",'bus_cnt :',bus_cnt, 'n_bus_cnt:',n_bus_cnt)
                    
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
                
    ############### 조이스틱 조작 모드 ###############
    else:
        error_I=0 # pid 제어에서 사용한 에러 누적값 초기화
        error_prev=0 
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

    # 사진 수집 버튼 인식
    cam_save_path = Path(cam.save_path)
    parent_folder = cam_save_path.name
    if collecting:
        ret, frame = cam.cap[0].read()
        if ret:
            timestamp = datetime.datetime.now().strftime('%Y%m%d')
            frame_path = cam.save_path / f"{parent_folder}_{cnt}.jpg"
            cnt += 1
            #중앙선 예측 위치 점 찍기
            cv2.circle(frame, (int(x),int(y)),10,(255,0,0),-1)
            #사용중인 모델 이름 남기기
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0,255,0)
            thickness = 2
            line_type = cv2.LINE_8
            text_position = (50,50)
            text = f"Model : {model_num}, No Bus Sign : {n_bus_cnt}, No Crosswalk Sign : {n_cross_cnt}"
            cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness, line_type)
            draw_boxes(frame, label, x1, y1, x2, y2)
            cv2.imwrite(str(frame_path), frame)  # 이미지 저장
            print(f"Saved frame at {timestamp}")
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
    
    # end_time = time.time()  # 루프 끝 시간 기록
    # loop_duration = end_time - start_time

    #if not gain_Tuning:
        #print(f"Loop duration: {loop_duration:.4f} seconds, steering input: {steering_input:.4f}, x: {x},y: {y}")
    
camera.cap[0].release()
cv2.destroyAllWindows()
pygame.quit()     

