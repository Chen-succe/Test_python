"""
author: chenjianyi
creat time: 2020-06-09 15:11
"""

import os
import cv2
from tqdm import tqdm
import torch
import tvm
import time
import dlib
import numpy as np
import argparse
from imutils import face_utils

import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='train')
    args = parser.parse_args()

    mode = args.mode
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/mnt/mfs2/haoyu.wang/project/dlib/shape_predictor_81_face_landmarks.dat")
    train_txt = f'/mnt/mfs2/haoyu.wang/project/repvgg_norm/huokai_{mode}_PIL_resize224.txt'
    output_dir = f'/data/data1/haoyu.wang/huohua_resize/{mode}_PIL_landmarks'
    os.makedirs(output_dir, exist_ok=True)
    no_face = 0
    txt = []
    with open(train_txt, 'r') as f:
        for k, l in enumerate(tqdm(f.readlines())):
            strs = l.strip().split('<blank>')
            img_path = strs[0]
            name = os.path.basename(strs[0]).split('.')[0] + '.npy'
            # img = dlib.load_rgb_image(img_path)
            img = cv2.imread(img_path)[:,:,::-1]
            # 定义人脸关键点检测器
            # 检测得到的人脸
            faces = detector(img, 1)
            # 如果存在人脸
            if len(faces):
                for i in range(len(faces)):
                    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
                    np.save(os.path.join(output_dir,name),landmarks)
            else:
                print(img_path," not find face")
                no_face+=1
    print(f'Find no face: {no_face}')







    # print(boxes)
    # print(landmarks)

