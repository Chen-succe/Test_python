# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, IterableDataset
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
sys.path.append('../')
import albumentations as alb
from torchtoolbox.transform import Cutout
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

class Loc_Dataset(Dataset):
    def __init__(self, data_file, phase='train',image_size=224):
        assert phase in ['train', 'val', 'test']
        image_list,label_list = [],[]
        with open(data_file, 'rt') as f:
            for line in f.readlines():
                strs = line.strip().split('<blank>')
                image_list.append(strs[0])
                label_list.append(int(strs[1]))
        path_lm = f'/{phase}_PIL_landmarks/'
        path_img = f'/{phase}_PIL/'
        # label_list = [label_list[i] for i in range(len(image_list)) if os.path.isfile(
        #     image_list[i].replace(path_img, path_lm).replace('.png', '.npy'))]
        # image_list = [image_list[i] for i in range(len(image_list)) if os.path.isfile(
        #     image_list[i].replace(path_img, path_lm).replace('.png', '.npy'))]
        self.path_lm = path_lm
        self.path_img = path_img
        # print(f'LOC({phase}): {len(image_list)}')
        self.image_list = image_list
        self.label_list = label_list
        self.transform = self.get_last_transforms(image_size)
        self.strong_transform = self.get_strong_transforms()
        self.image_size = (image_size, image_size)
        self.phase = phase
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        flag = True
        while flag:
            try:
                filename = self.image_list[idx]
                label = self.label_list[idx]
                img = np.array(Image.open(filename))
                landmark = None
                if os.path.exists(filename.replace('.png', '.npy').replace(self.path_img, self.path_lm)):
                    landmark = np.load(filename.replace('.png', '.npy').replace(self.path_img, self.path_lm))#[0]
                    landmark = self.reorder_landmark(landmark)
                mask = self.gen_mask(img,landmark)
                if self.phase == 'train':
                    transformed = self.strong_transform(image=img, mask= mask)
                    img = transformed['image']
                    mask = transformed['mask']
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
                mask = mask.permute(2,0,1).to(torch.float32)
                flag = False
            except Exception as e:
                print(e)
                idx = torch.randint(low=0, high=len(self)-1, size=(1,)).item()
        return img,mask,label
    def get_strong_transforms(self):
        return alb.Compose([
            alb.HorizontalFlip(),
            alb.OneOf([
                alb.VerticalFlip(),
                alb.Transpose(),
                alb.ShiftScaleRotate()
            ]),
            alb.SomeOf([
                alb.GaussNoise(),
                alb.CLAHE(),
                alb.ChannelShuffle(),
                alb.ColorJitter(),
                alb.HueSaturationValue(),
                alb.RandomBrightnessContrast(),
            ],n=2,p=0.6),
            # ToSepia(p=0.1),
            alb.RandomGridShuffle([2, 2], p=0.2),
            alb.ImageCompression(quality_lower=30, quality_upper=70)
        ])
    def get_last_transforms(self,size,):
        # return   transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((size, size)),
        #     Cutout(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.560, 0.449, 0.407], std=[0.248, 0.229, 0.223])
        # ])
        return alb.Compose([
            alb.Resize(size,size),
            alb.Normalize(mean=[0.560, 0.449, 0.407], std=[0.248, 0.229, 0.223]),
            ToTensorV2()
            ])
    def gen_mask(self, img, landmark):
        if landmark is not None:
            eye_mask = np.zeros_like(img[:, :, 0])
            eye_landmark = np.concatenate([landmark[17:27],landmark[36:48]])
            cv2.fillConvexPoly(eye_mask, cv2.convexHull(eye_landmark), 1.)
            nose_mask = np.zeros_like(img[:, :, 0])
            nose_landmark =landmark[27:36]
            cv2.fillConvexPoly(nose_mask, cv2.convexHull(nose_landmark), 1.)
            mouse_mask = np.zeros_like(img[:, :, 0])
            mouse_landmark = landmark[48:60]
            cv2.fillConvexPoly(mouse_mask, cv2.convexHull(mouse_landmark), 1.)
            mask = np.zeros_like(img[:, :, 0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)
            background_mask =  np.ones_like(img[:, :, 0])
            background_mask[mask==1] = 0
            mask[eye_mask==1] = 0
            mask[nose_mask==1] = 0
            mask[mouse_mask==1] = 0
            return np.stack([mask,eye_mask,nose_mask, mouse_mask,background_mask],axis=-1)
        return np.ones((img.shape[0],img.shape[1],5))

    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark
    def worker_init_fn(self,worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_dataset = Loc_Dataset(data_file= '/mnt/mfs2/haoyu.wang/project/repvgg_norm/data_txts/huokai_train_PIL_resize224.txt',phase='train', image_size=224)
    print(len(image_dataset[0]))
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(image_dataset,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=0,
                                             worker_init_fn=image_dataset.worker_init_fn)
    for data in dataloader:
        break

    # print(data)
    print(len(data))
    print(data[0].shape)
    print(data[1].shape)
    # print(data[0])
    # print(data[1])
    # img = data['img']
    # img = img.view((-1, 3, 224, 224))
    utils.save_image(data[0], 'loader.png', nrow=batch_size, normalize=True, range=(0, 1))
    for i in range(5):
        utils.save_image(data[1][:,i:i+1,:,:].to(torch.float32), f'mask_{i}.png', nrow=batch_size, normalize=False, range=(0, 1))