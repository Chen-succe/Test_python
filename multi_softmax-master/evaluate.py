import os.path
import glob
import argparse
import random
from tqdm import tqdm
from PIL import Image
import torch
from util.datasets import get_transform
from repvgg.repvgg import create_RepVGG_B0
import torch.nn.functional as F
from util import CustomDataset
from torch.utils.data import DataLoader
from imagecorruptions import corrupt, get_corruption_names
import numpy as np
import cv2


# def dg(image, corruption_name,severity):
#     image = np.array(image)
#     image = corrupt(image, corruption_name=corruption_name, severity=severity)
#     image = Image.fromarray(image)
#     return image
class FaceModel(torch.nn.Module):
    """
    Attributes
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, num_classes):
        """Init face model by backbone factorcy and head factory.
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = create_RepVGG_B0(num_classes=0)
        self.head = torch.nn.Linear(1280, num_classes)
        torch.nn.init.xavier_uniform_(self.head.weight)
        self.head_binary = torch.nn.Linear(1280, 2)
        torch.nn.init.xavier_uniform_(self.head_binary.weight)

    def forward(self, data):
        feat = self.backbone.forward(data)
        feat = F.dropout(feat, training=self.training)
        pred = self.head(feat)
        output = self.head_binary(feat)
        return pred, output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='', type=str)
    parser.add_argument('--th', type=float)
    parser.add_argument('--gpus', default='')
    parser.add_argument('--multi_head', action='store_true', default=True)
    parser.add_argument('--txt', default='data_txts/all_val_PIL_resize224.txt', type=str)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--num_classes', default=40, type=int)
    args = parser.parse_args()
    weight = args.weight
    size = args.size
    th = args.th
    assert weight != "" and th != None
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('Building Model')

    model = FaceModel(args.num_classes)
    ckpt = torch.load(weight, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model = model.to(device)
    transform_test = get_transform('test', size)
    val_dataset = CustomDataset(args.txt, type='test', transforms=transform_test)
    imgs = []
    with open(args.txt, 'r') as f:
        for line in f.readlines():
            strs = line.strip().split('<blank>')
            if int(strs[1]) == 0:
                continue
            imgs.append((strs[0], int(strs[1])))
    val_dataset.imgs = imgs
    batch_size = 512 if size == 224 else 128
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=16,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True
                            )

    total_num = 0
    acc = 0
    TN = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluate'):
            images, labels = images.to(device), labels.to(device)
            bs = images.shape[0]
            preds, output = model(images)
            preds = preds.argmax(1)
            probs = F.softmax(output, dim=1)
            score = probs[:, 1]
            TN += (score > th).sum().item()
            preds = preds[score > th]
            labels = labels[score > th]
            total_num += bs
            acc += preds.eq(labels).sum().item()
    print(f'Fake acc@multi: {acc / total_num * 100:.2f}')
    print(f'Fake TNR@: {TN / total_num * 100:.2f}')

    # pos = []
    # neg = []
    # # random.shuffle(imgs)
    # with torch.no_grad():
    #
    #
    #
    #
    #     for img_path,l in tqdm(imgs):
    #         if l!=12 and l!=0:
    #             continue
    #         img = Image. open(img_path)
    #         input = transform_test(img).to(device)
    #         inputs = [input]
    #         img_name = os.path.basename(img_path)
    #         name= ['Origin']
    #         for crp_name in ['elastic_transform','pixelate']:
    #             for level in range(5):
    #                 crp_img = dg(img,crp_name,level+1)
    #                 crp_img.save(f'visualization/corruption/{img_name[:-4]}_{crp_name}_{level}.png')
    #                 input = transform_test(crp_img).to(device)
    #                 inputs.append(input)
    #                 name.append(f"{crp_name}_{level+1}")
    #         inputs = torch.stack(inputs)
    #         output, binary_output = model(inputs,0)
    #         if args.multi_head:
    #             prob = torch.softmax(binary_output,dim=1)
    #         else:
    #             real_dist = output[:, :1]
    #             fake_dist = output[:, 1:]
    #             real_dist = torch.mean(real_dist, dim=1, keepdim=True)
    #             fake_dist, _ = torch.max(fake_dist, dim=1, keepdim=True)
    #             dist = torch.cat((real_dist, fake_dist), dim=1)
    #             prob = torch.softmax(dist * model.head.scale, dim=1)
    #
    #         score = prob[:,1]
    #         # score[score>th] = 1
    #         # score[score<=th] = 0
    #         print("label:",l)
    #         # print('Corruption:',name)
    #         print("Score:",score.tolist())
    #     #     if score > th:
    #     #         name = os.path.basename(img_path)
    #     #         # img.save(os.path.join(output_dir,name.replace('jpg','png')))
    #     #         # print(name,prob)
    #     #         neg.append(img_path)
    #     #     else:
    #     #         pos.append(img_path)
    #     # print(f'FRR: {len(neg)/total_nums:.4f}')
    #
