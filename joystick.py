import os
import pygame
import cv2
import logging
import datetime
from typing import Sequence
from jetracer.nvidia_racecar import NvidiaRacecar
import sys
from pathlib import Path

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
frame_rate = 30
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


save_path = str('record')
# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

car.throttle_gain = 0.4

joystick = pygame.joystick.Joystick(0)
joystick.init()

running = True
throttle_range = (-0.5, 0.5)


# 조이스틱 버튼 플래그
flag_4_1=False
flag_4_2=False
flag_0_1=False
flag_0_2=False
#1번 : 카메라 녹화 버튼
flag_1_1=False
flag_1_2=False
#6번 : 스티어링 왼쪽 보정
flag_6_1=False
flag_6_2=False
#7번 : 스티어링 오른쪽 보정
flag_7_1=False
flag_7_2=False
collecting = False
cnt = 1
while running:
    pygame.event.pump()

    throttle = -joystick.get_axis(1)
    throttle = max(throttle_range[0], min(throttle_range[1], throttle))
    car.throttle = throttle
    steering = joystick.get_axis(2)
    car.steering = steering-0.296


    # print("thro: {:2f} steer: {:2f} th_gain: {:2f} ".format(throttle, steering,car.throttle_gain))
    if joystick.get_button(11): # start button
        running = False


    # Throttle Gain 증가
    flag_4_1 = joystick.get_button(4)
    if flag_4_1 and not flag_4_2:
        car.throttle_gain+=0.02
    flag_4_2=flag_4_1   

    # Throttle Gain 감소
    flag_0_1 = joystick.get_button(0)
    if flag_0_1 and not flag_0_2:
        car.throttle_gain-=0.02
    flag_0_2=flag_0_1

    # 스티어링 왼쪽 보정
    flag_6_1 = joystick.get_button(6)
    if flag_6_1 and not flag_6_2:
        car.steering = -0.3
        print("steering left")
        print("steering gain :", car.steering_gain)
        print("steering :", car.steering)
        print("steering offset:", car.steering_offset)
    flag_6_2=flag_6_1   

    # 스티어링 오른쪽 보정
    flag_7_1 = joystick.get_button(7)
    if flag_7_1 and not flag_7_2:
        car.steering = 0.3
        print("steering right")
        print("steering gain :", car.steering_gain)
        print("steering :", car.steering)
        print("steering offset:", car.steering_offset)
    flag_7_2=flag_7_1

    #gstreamer 파이프라인 설정
    sensor_id = 0
    downscale = 2
    width, height = (1280, 720)
    _width, _height = (width // downscale, height // downscale)
    frame_rate = 30
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
    #녹화 
    if joystick.get_button(1) and not flag_1_1:
        collecting = not collecting
        flag_1_1 = True
        if collecting:
            print("Starting image Collection...")
        else:
            print("Stopping image Collection...")
            
    elif not joystick.get_button(1) and flag_1_1:
        flag_1_1 = False

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
camera.cap[0].release()
cv2.destroyAllWindows()
pygame.quit()        
