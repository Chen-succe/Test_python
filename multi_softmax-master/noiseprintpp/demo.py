"""
@Author : Chen Peng
@Date  : 2023/7/7 18:00
"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from DnCNN import make_net
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NoiseprintPP:
    def __init__(self, weights_path='weights/nosieprint++.pth', device='cpu'):
        super(NoiseprintPP, self).__init__()
        self.weights_path = weights_path
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        # device = torch.device('cuda:{}'.format(self.gpu_id))
        num_levels = 17
        out_channel = 1
        dncnn = make_net(3, kernels=[3, ] * num_levels,
                         features=[64, ] * (num_levels - 1) + [out_channel],
                         bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                         acts=['relu', ] * (num_levels - 1) + ['linear', ],
                         dilats=[1, ] * num_levels,
                         bn_momentum=0.1, padding=1)
        checkpoint = torch.load(self.weights_path, map_location='cpu')
        dncnn.load_state_dict(checkpoint)
        dncnn.to(self.device)
        dncnn.eval()
        return dncnn

    def get_noiseprint(self, img_pth):
        img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
        img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float) / 256.0

        img = img.unsqueeze(0)
        with torch.no_grad():
            noiseprint = self.model(img)
        noiseprint = noiseprint.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return noiseprint

    def draw_noiseprint(self, img_pth, save_pth):
        noiseprint = self.get_noiseprint(img_pth)
        plt.imshow(noiseprint[16:-16, 16:-16], cmap='gray')
        plt.axis('off')
        plt.savefig(save_pth, bbox_inches='tight', pad_inches=0)

    def draw_noiseprint_for_video(self, video_pth, save_pth):
        # Draw noiseprint for video. For each frame, we draw a noiseprint with cmap='gray'
        # and save all frames as video.
        # video_pth: the path of video
        # save_pth: the path of saving noiseprint
        cap = cv2.VideoCapture(video_pth)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_pth, fourcc, fps, (width, height))
        for _ in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float) / 256.0
                img = img.unsqueeze(0)
                img = img.cuda(self.gpu_id)
                with torch.no_grad():
                    noiseprint = self.model(img)
                noiseprint = noiseprint.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                # tile last axis to 3 channels
                noiseprint = np.tile(noiseprint, (1, 1, 3))
                noiseprint = np.array(noiseprint * 255, dtype=np.uint8)
                out.write(noiseprint)
        cap.release()

    def draw_noiseprint_with_image(self, img_pth, save_pth):
        img = np.array(Image.open(img_pth).convert("RGB"))
        img_gray = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
        img_mid = cv2.medianBlur(img_gray, 3)
        noiseprint = self.get_noiseprint(img_pth)
        noise_mid = np.abs(img_gray - img_mid)

        plt.subplot(131)
        plt.imshow(img)
        plt.axis('off')
        plt.title('RGB', fontsize=10)
        plt.subplot(132)
        plt.imshow(noise_mid)
        plt.axis('off')
        plt.title('median', fontsize=10)
        plt.subplot(133)
        plt.imshow(noiseprint[16:-16, 16:-16], cmap='gray')
        plt.axis('off')
        plt.title('noiseprint++', fontsize=10)
        plt.savefig(save_pth, bbox_inches='tight', pad_inches=0)


def get_file_list(root_path):
    file_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


import random


class NoiseDataset(Dataset):
    def __init__(self, data_file, transforms=None):
        self.transforms = transforms
        self.imgs = []
        if data_file is not None:
            f = open(data_file, 'rt')
            for line in f.readlines():
                strs = line.strip().split('<blank>')
                self.imgs.append((strs[0], int(strs[1])))
            f.close()

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        data = None
        while data is None:
            img_path, label = self.imgs[index]
            try:
                data = Image.open(img_path).convert('RGB')
            except:
                print(img_path)
                pass
            index = random.randint(0, len(self.imgs) - 1)
        if self.transforms is not None:
            data = self.transforms(data)
        return data, label, os.path.basename(img_path)

    def __len__(self):
        return len(self.imgs)

    def worker_init_fn(self, worker_id):
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', default='../data_txts/huokai_train_PIL_resize224.txt', type=str)
    parse.add_argument('--gpus', default='')
    parse.add_argument('--bs', default=128, type=int)
    parse.add_argument('--serve', default='8g4')
    args = parse.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.serve == '8g4':
        output_dir = f'/data/data1/haoyu.wang/qita_neg/swhc'

    elif args.serve == '8g8':
        output_dir = f'/data/haoyu.wang/huohua_resize/test_PIL_noiseprint'

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    dncnn = NoiseprintPP(device=device)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if os.path.isfile(args.path):
        dataset = NoiseDataset(args.path, transform)
        dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=8, pin_memory=True)
        for images, _, paths in tqdm(dataloader):
            images = images.to(device)
            with torch.no_grad():
                noiseprint = dncnn.model(images)
            noiseprint = noiseprint.cpu().numpy().transpose(0, 2, 3, 1)
            for noise, name in zip(noiseprint, paths):
                # noise = np.uint8(((noise-noise.min())/(noise.max()-noise.min()))*255.)
                # cv2.imwrite(os.path.join(output_dir,name),noise*255)
                np.save(os.path.join(output_dir, name.split('.')[0] + '.npy'), noise)
    elif os.path.isdir(args.path):
        for path in ["fdd_21_1", "hz_22_注入", "nx_21_1", "tx_22_1", "zy_21_1", "zy_22_1"]:  #
            output_dir_ = os.path.join(output_dir, path)
            os.makedirs(output_dir_, exist_ok=True)
            # print('Building Datasets: ',path)
            import glob

            images_list = glob.glob(os.path.join(args.path + path, '*'))
            imgs = []
            for img_path in images_list:
                imgs.append([img_path, 1])
            dataset = NoiseDataset(None, transform)
            dataset.imgs = imgs
            dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=8, pin_memory=True)
            for images, _, paths in tqdm(dataloader):
                images = images.to(device)
                with torch.no_grad():
                    noiseprint = dncnn.model(images)
                noiseprint = noiseprint.cpu().numpy().transpose(0, 2, 3, 1)
                for noise, name in zip(noiseprint, paths):
                    noise = np.uint8(((noise - noise.min()) / (noise.max() - noise.min())) * 255.)
                    cv2.imwrite(os.path.join(output_dir_, name), noise)
                    # np.save(os.path.join(output_dir_, name.split('.')[0] + '.npy'), noise)
