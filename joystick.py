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

sys.path.append('/home/ircv6/HYU-2024-Embedded/week07/Camera')
import camera
cam = camera.Camera(
        sensor_id = 0,
        window_title = 'Camera',
        save_path = 'record',
        save = True,
        stream = False,
        log = False)

car = NvidiaRacecar()

#gstreamer 파이프라인 설정
sensor_id = 0
downscale = 2
width, height = (1280, 720)
_width, _height = (width // downscale, height // downscale)
frame_rate = 2
flip_method = 0
contrast = 1.3
brightness = 0.2

gstreamer_pipeline = (
    "nvarguscamerasrc sensor-id=%d ! "
    "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
    "nvvidconv flip-method=%d, interpolation-method=1 ! "
    "videobalance contrast=%.1f brightness=%.1f ! "
    "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
    % (
        sensor_id,
        width,
        height,
        frame_rate,        
        flip_method,
        contrast,
        brightness,
        _width,
        _height,
    )
)


model = models.alexnet(num_classes=2, dropout=0.0)
device = torch.device('cuda')
model.load_state_dict(torch.load('/home/ircv6/HYU-2024-Embedded/jetracer/road_following_model_straight.pth'))
model = model.to(device)

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


# 조이스틱 버튼 플래그
flag_4_1=False
flag_4_2=False

flag_0_1=False
flag_0_2=False

# 1번 : 카메라 녹화 버튼
flag_1_1=False
flag_1_2=False

# 3번 : 게인 모드에서 조절 버튼
flag_3_1=False
flag_3_2=False

# 6번 : 자율 주행
flag_6_1=False
flag_6_2=False

#7번 : Throttle / 제어 Gain 설정 변경 
flag_7_1=False
flag_7_2=False

automode = False
collecting = False
cnt = 0
gain_Tuning = False
gain_number = 0
gain_name = "Kp"

## 제어 파라미터
th_control=0.37
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



while running:
    pygame.event.pump()
    #자율 주행 모드
    if automode:
        ret, frame = cam.cap[0].read()
        # throttle = 0.4
        # throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        # car.throttle = throttle
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

                ## 조향 제어 로직(pid)
                reference=x-445
                steering_input = steering_control(reference,K_p,K_i,K_d,K_a)
                car.steering = steering_input
                

                # 모니터링
                if gain_number==0:
                    gain_name="Kp"
                elif gain_number==1:
                    gain_name="Ki"
                elif gain_number==2:
                    gain_name="Kd"
                else:
                    gain_name="Ka"
                
                if gain_Tuning: #튜닝 모드로 작동
                    car.throttle = 0
                    print("Tuning mode, gain_num: {} K_p: {}, K_i: {} K_d: {} K_a: {} x_error: {} y_axis: {} ".format(gain_name, K_p, K_i,K_d,K_a,reference,y))
                else: # 드라이빙 모드로 작동
                    car.throttle = th_control
                    print("Driving mode, thro: {:2f} steer input: {} K_p: {} x_error: {} y_axis: {} ".format(th_control, steering_input,K_p,reference,y))
                
                # # 기존 제어 로직
                # car.throttle=th_control
                # error_feedback = x-640
                # if error_feedback > deadzone:
                #     car.steering = 0.7
                #     #car.throttle=0.35
                #     print("Car is on left side")
                # elif error_feedback < -deadzone:
                #     car.steering = -1
                #     #car.throttle=0.32
                #     print("Car is on right side")
                # else:
                #     car.steering = -0.3
                #     #car.throttle=0.32
                #     print("Car is on center line")

                #print(x, y,"Steering input: {}".format(steering_input()))

    #조이스틱으로 조작
    else:
        error_I=0 # pid 제어에서 사용한 에러 누적값 초기화
        error_prev=0 
        throttle = -joystick.get_axis(1)
        throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        car.throttle = throttle
        steering = joystick.get_axis(2)
        car.steering = steering
        # print("thro: {:2f} steer input: {} ".format(throttle, steering))
    
    print("thro: {:2f} steer: {:2f} th_gain: {:2f} deadzone: {} ".format(throttle, car.steering_gain*steering,car.throttle_gain,deadzone))
    
    
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

    # # Throttle Gain 증가
    # flag_4_1 = joystick.get_button(4)
    # if flag_4_1 and not flag_4_2:
    #     car.throttle_gain += 0.02
    # flag_4_2=flag_4_1   

    # # Throttle Gain 감소
    # flag_0_1 = joystick.get_button(0)
    # if flag_0_1 and not flag_0_2:
    #     car.throttle_gain -= 0.02
    # flag_0_2=flag_0_1

    # # Deadzone 증가
    # flag_4_1 = joystick.get_button(4)
    # if flag_4_1 and not flag_4_2:
    #     deadzone += 10
    #     deadzone = min(640,deadzone)
    # flag_4_2=flag_4_1   

    # # Deadzone 감소
    # flag_0_1 = joystick.get_button(0)
    # if flag_0_1 and not flag_0_2:
    #     deadzone-= 10
    #     deadzone = max(0,deadzone)
    # flag_0_2=flag_0_1

    #gstreamer 파이프라인 설정
    sensor_id = 0
    downscale = 2
    width, height = (1280, 720)
    _width, _height = (width // downscale, height // downscale)
    frame_rate = 2
    flip_method = 0
    contrast = 1.3
    brightness = 0.2

    gstreamer_pipeline = (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d, interpolation-method=1 ! "
        "videobalance contrast=%.1f brightness=%.1f ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            width,
            height,
            frame_rate,        
            flip_method,
            contrast,
            brightness,
            _width,
            _height,
        )
    )

    #사진 수집이 활성화된 경우
    cam_save_path = Path(cam.save_path)
    parent_folder = cam_save_path.name
    if collecting:
        ret, frame = cam.cap[0].read()
        if ret:
            timestamp = datetime.datetime.now().strftime('%Y%m%d')
            frame_path = cam.save_path / f"{parent_folder}_{cnt}.jpg"
            cnt += 1
            cv2.imwrite(str(frame_path), frame)  # 이미지 저장
            print(f"Saved frame at {timestamp}")
        else:
            print("Failed to capture frame")
    
    # 자율주행
    if joystick.get_button(6) and not flag_6_1:
        automode = not automode
        flag_6_1 = True
        if automode:
            print("AutoDriving Start...")
        else:
            print("AudoDriving Stopped...")
    elif not joystick.get_button(6) and flag_6_1:
        flag_6_1 = False
    
camera.cap[0].release()
cv2.destroyAllWindows()
pygame.quit()     
