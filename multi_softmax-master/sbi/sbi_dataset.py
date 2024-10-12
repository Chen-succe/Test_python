# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, IterableDataset
from sbi.func import IoUfrom2bboxes, crop_face, RandomDownScale
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
sys.path.append('../')
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')
import sbi.blend as B


class SBI_Dataset(Dataset):
    def __init__(self, data_file, phase='train', image_size=224):
        assert phase in ['train', 'val', 'test']
        image_list,label_list = [],[]
        with open(data_file, 'rt') as f:
            for line in f.readlines():
                strs = line.strip().split('<blank>')
                image_list.append(strs[0])
                label_list.append(int(strs[1]))
        path_lm = f'/{phase}_PIL_landmarks/'
        path_img = f'/{phase}_PIL/'
        label_list = [label_list[i] for i in range(len(image_list)) if os.path.isfile(
            image_list[i].replace(path_img, path_lm).replace('.png', '.npy'))]
        image_list = [image_list[i] for i in range(len(image_list)) if os.path.isfile(
            image_list[i].replace(path_img, path_lm).replace('.png', '.npy'))]
        self.path_lm = path_lm
        self.path_img = path_img
        print(f'SBI({phase}): {len(image_list)}')
        self.image_list = image_list
        self.label_list = label_list
        self.image_size = (image_size, image_size)
        self.phase = phase
        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.last_transforms = self.get_last_transforms(image_size)
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        flag = True
        while flag:
            try:
                filename = self.image_list[idx]
                label = self.label_list[idx]
                img = np.array(Image.open(filename))
                landmark = np.load(filename.replace('.png', '.npy').replace(self.path_img, self.path_lm))#[0]
                # bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
                # bboxes = np.load(filename.replace('.png', '.npy').replace('/frames/', '/retina/'))[:2]
                bbox = None
                # iou_max = -1
                # for i in range(len(bboxes)):
                #     iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
                #     if iou_max < iou:
                #         bbox = bboxes[i]
                #         iou_max = iou

                landmark = self.reorder_landmark(landmark)
                if self.phase == 'train':
                    if np.random.rand() < 0.5:
                        img, _, landmark, bbox = self.hflip(img, None, landmark, bbox)

                # img, landmark, bbox, __ = crop_face(img, landmark, bbox, margin=True, crop_by_bbox=False)
                img_r, img_f, mask_f = self.self_blending(img.copy(), landmark.copy())
                if self.phase == 'train':
                    transformed = self.transforms(image=img_f.astype('uint8'), image1=img_r.astype('uint8'))
                    img_f = transformed['image']
                    img_r = transformed['image1']
                # img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(img_f, landmark, bbox, margin=False,
                #                                                               crop_by_bbox=True, abs_coord=True,
                #                                                               phase=self.phase)
                # img_r = img_r[y0_new:y1_new, x0_new:x1_new]
                img_f = cv2.resize(img_f, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32') / 255
                img_r = cv2.resize(img_r, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32') / 255
                img_f = torch.from_numpy(img_f.transpose((2, 0, 1)))
                img_r = torch.from_numpy(img_r.transpose((2, 0, 1)))
                # transformed = self.last_transforms(image=img_f, image1=img_r)
                # img_f = transformed['image']
                # img_r = transformed['image1']
                flag = False
            except Exception as e:
                print(e)
                idx = torch.randint(low=0, high=len(self)-1, size=(1,)).item()

        return img_f, img_r,1,label

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.3, 0.3), p=1),
                alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
            ], p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),

        ], p=1.)

    def get_transforms(self):
        return alb.Compose([

            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

        ],
            additional_targets={f'image1': 'image'},
            p=1.)
    def get_last_transforms(self,size):
        return alb.Compose([
            alb.Resize(size,size),
            alb.Normalize(mean=[0.560, 0.449, 0.407], std=[0.248, 0.229, 0.223]),
            ToTensorV2()
            ],
            additional_targets={f'image1': 'image'},
            )

    def randaffine(self,img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)
        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )
        transformed = f(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def self_blending(self, img, landmark):
        H, W = len(img), len(img[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]

        mask = np.zeros_like(img[:, :, 0])
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)

        img_blended, mask = B.dynamic_blend(source, img, mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img, img_blended, mask

    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self,img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        if landmark is not None:
            landmark_new = np.zeros_like(landmark)
            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]
            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]
            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]
            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]
            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]
            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W - landmark_new[:, 0]
        else:
            landmark_new = None
        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W - bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W - bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W - bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W - bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W - bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W - bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None
        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new

    def collate_fn(self, batch):

        img_f, img_r,label_1,label = zip(*batch)
        images = torch.cat([torch.stack(img_r), torch.stack(img_f)], 0)
        labels = torch.cat([torch.tensor(label),torch.tensor(label_1)],0)
        return images,labels

    def worker_init_fn(self,worker_id):
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_dataset = SBI_Dataset(data_file= '/mnt/mfs2/haoyu.wang/project/repvgg_norm/data_txts/huokai_train_PIL_resize224.txt',phase='train', image_size=224)
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(image_dataset,
                                             batch_size=4,
                                             shuffle=True,
                                             collate_fn=image_dataset.collate_fn,
                                             num_workers=0,
                                             worker_init_fn=image_dataset.worker_init_fn
                                             )
    data_iter = iter(dataloader)
    data = next(data_iter)
    # print(data)
    print(len(data))
    print(data[0])
    # print(data[1].shape)
    print(data[1])
    # img = data['img']
    # img = img.view((-1, 3, 224, 224))
    utils.save_image(data[0], 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))