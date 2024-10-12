# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from PIL import Image
import cv2
from torch.utils import data
from torchvision import transforms as T
import random
import numpy as np
from torchtoolbox.transform import Cutout
import lmdb
import pickle
from albumentations import ( RandomResizedCrop,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,VerticalFlip,
    Transpose, ShiftScaleRotate, Blur,GaussianBlur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, SomeOf,Compose,RandomBrightness,ToSepia, ImageCompression, ColorJitter,RandomGridShuffle,ChannelShuffle
)
def get_transform(mode='train',resize = 224):
    if mode=='train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop((resize, resize)),
            #transforms.Resize((resize, resize)),
            Cutout(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.560, 0.449, 0.407], std=[0.248, 0.229, 0.223])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            #transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.560, 0.449, 0.407], std=[0.248, 0.229, 0.223])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
    return transform

def strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(),
        OneOf([
            VerticalFlip(),
            Transpose(),
            ShiftScaleRotate()
        ]),
        # OneOf([
        #     GaussNoise(),
        # ], p=0.5),
        # OneOf([
        #     MotionBlur(p=0.25),
        #     GaussianBlur(p=0.5),
        #     Blur(blur_limit=3, p=0.25),
        # ], p=0.2),
        # HueSaturationValue(p=0.2),
        SomeOf([
            # RandomBrightness(),
            # Sharpen(),
            # Emboss(),
            GaussNoise(),
            CLAHE(),
            ChannelShuffle(),
            ColorJitter(),
            HueSaturationValue(),
            RandomBrightnessContrast(),
        ],n=2,p=0.6),
        # ToSepia(p=0.1),
        RandomGridShuffle([2,2],p=0.2),
        ImageCompression(quality_lower=30, quality_upper=70)
    ], p=p)

def origin_strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(),
        OneOf([
            GaussNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.25),
            GaussianBlur(p=0.5),
            Blur(blur_limit=3, p=0.25),
        ], p=0.2),
        HueSaturationValue(p=0.2),
        OneOf([
            RandomBrightness(),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast(),
        ], p=0.6),
        ToSepia(p=0.1),
        ImageCompression(quality_lower=10, quality_upper=70)
    ], p=p)

augmentation=strong_aug()

def strong_transform_train(image):
    image = np.array(image)
    image = augmentation(image=image)['image']
    # return image
    return image

def identity(x):
    return x

class CustomDataset(data.Dataset):
    def __init__(self, data_file, type, transforms=None):
        self.transforms = transforms
        self.type = type
        self.imgs = []
        # self.env = lmdb.open(data_file, readonly=True, lock=False, readahead=False, meminit=False)
        # meta_info = pickle.load(open(os.path.join(data_file, 'meta_info.pkl'), "rb"))
        # self.resolution = meta_info['resolution']
        # self.keys = meta_info['keys']
        # self.labels = meta_info['label']

        if data_file is not None:
            f = open(data_file, 'rt')
            for line in f.readlines():
                strs = line.strip().split('<blank>')
                self.imgs.append((strs[0], int(strs[1])))
            f.close()


    def __getitem__(self, index):
        # print(f'index :{index}',np.random.randint(1,100,3))
        """
        一次返回一张图片的数据
        """
        # while(1):
        data = None
        while data is None:
            # read one image
            # key = self.keys[index]
            # label = self.labels[index]
            # try:
            #     with self.env.begin(write=False) as txn:
            #         buf = txn.get(key.encode('ascii'))
            #     img_flat = np.frombuffer(buf, dtype=np.uint8)
            #     C, H, W = [int(s) for s in self.resolution[index].split('_')]
            #     data = img_flat.reshape(H, W, C)
            # except:
            #     print(key)
            #     pass
            img_path, label = self.imgs[index]
            try:
                # print(img_path)
                #data = Image.open(img_path).convert('RGB')
                data = cv2.imread(img_path)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                data = cv2.resize(data, (224, 224))
            except:
                print(img_path)
                pass
            index = random.randint(0,len(self.imgs)-1)
        if self.type == 'train':
            data = strong_transform_train(data)
        if self.transforms is not None:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

    def worker_init_fn(self,worker_id):
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_dataset = CustomDataset(data_file='/mnt/mfs2/haoyu.wang/project/repvgg_norm/data_txts/huokai_train_PIL_resize224_8g8.txt', type='train',
        transforms=get_transform('train',224))

    batch_size = 4
    dataloader = torch.utils.data.DataLoader(image_dataset,
                                             batch_size=8,
                                             shuffle=True,
                                             num_workers=0,
                                             worker_init_fn=image_dataset.worker_init_fn)
    for data in dataloader:
        break
    # print(data)
    print(len(data))
    print(data[0].shape)
    print(data[1].shape)
    print(data[0])
    # print(data[1])
    # img = data['img']
    # img = img.view((-1, 3, 224, 224))
    from torchvision.utils import save_image
    save_image(data[0], 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
    # save_image(data[1][:, :1, :, :].to(torch.float32), 'mask.png', nrow=batch_size, normalize=False, range=(0, 1))
