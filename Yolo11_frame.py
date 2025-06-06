# -*- coding: cp932 -*-

from ultralytics import YOLO
import os
import random
import shutil
import numpy as np
import pandas as pd
import cv2
import yaml
import matplotlib.pyplot as plt
import glob
import mmap
import struct
import time
from sklearn.model_selection import train_test_split

DETECTION_STRUCT = 'ifiiii'  # class_id (int), conf (float), x1,y1,x2,y2 (int�~4)
MAX_DETECTIONS = 20  # �ő�20���܂ŋL�^
WIDTH = 640
HEIGHT = 480
CHANNELS = 3
HEADER_FORMAT = 'III'  # unsigned int �~ 3 = ���E�����E�`���l����
SHM_NAME = 'Local\\OpenCVFrame'
SHM_SIZE = WIDTH * HEIGHT * CHANNELS + struct.calcsize(HEADER_FORMAT) + 4 + MAX_DETECTIONS * struct.calcsize(DETECTION_STRUCT)

# ���L�������쐬
shm = mmap.mmap(-1, SHM_SIZE, tagname=SHM_NAME)

print("create shared memory")

#model = YOLO('yolo11n.pt')

#model.train(data="dataset.yaml", epochs=100, batch=8)


model = YOLO('../../../Yolo11_frame/runs/detect/runs_m/detect/train/weights/best.pt')
#"C:\Users\ogawa\desktop\visual_studiio_project\Automatic_alignment\Yolo11_frame\runs\detect\runs_m\detect\train\weights\best.pt"

# OpenCV��Web�J�������N��
cap = cv2.VideoCapture(0)  # 0�̓f�t�H���g�J����

while True:
    ret, frame = cap.read()
    if not ret:
        break
    start_time = time.time()
    # YOLOv8�Ő��_�icv2�摜�𒼐ړn���j
    results = model.predict(frame, conf=0.3, iou=0.5)

    detections = []
    

    # ���o���ʂ�`��
    for result in results:
        boxes = result.boxes
        # confidence�i�M���x�j�Ń\�[�g���čŏ�ʂ̂ݎ擾
        sorted_boxes = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)
        if len(sorted_boxes) > 0:
            box = sorted_boxes[0]  # �ł��M���x�̍������o����
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # ���W�𐮐���
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detections.append((cls_id, conf, x1, y1, x2, y2))

            # �o�E���f�B���O�{�b�N�X�ƃ��x����`��
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
            
    if len(detections) == 0:
                cv2.putText(frame,
                text='not detect',
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 0, 255),  # �ԕ���
                thickness=2,
                lineType=cv2.LINE_4)        

    # �E�B���h�E�ɕ\��
    end_time = time.time()
    fps = 1.0 / (end_time - start_time + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    header = struct.pack(HEADER_FORMAT, HEIGHT, WIDTH, CHANNELS)
    shm.seek(0)
    shm.write(header)
    shm.write(frame.tobytes())
    shm.write(struct.pack('I', len(detections)))

    # ���o�f�[�^�i�ő�MAX_DETECTIONS�܂Łj
    for det in detections:
        shm.write(struct.pack(DETECTION_STRUCT, *det))

    # ����Ȃ����̓p�f�B���O
    for _ in range(MAX_DETECTIONS - len(detections)):
        shm.write(struct.pack(DETECTION_STRUCT, -1, 0.0, 0, 0, 0, 0))
    #cv2.imshow("YOLOv8 Webcam Detection", frame)
    #print("frame")

    

# �I������
cap.release()
cv2.destroyAllWindows()