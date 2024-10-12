import shutil
from PIL import Image
import os
import os.path
import glob
import time
from shutil import copyfile
import random
from random import sample
import argparse
from tqdm import tqdm
import random
from repvgg.repvgg import create_RepVGG_B0
from PIL import Image
import torch
from head.AdM_Softmax import  ADM_Softmax
from util import get_transform,CustomDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
from models.xception import Xception
def resize_with_pad(image: np.array, new_shape,padding_color= (0, 0, 0)) -> np.array:
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])
    if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
        ratio = float(min(new_shape)) / min(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=padding_color)
    return image,ratio,(top,bottom,left,right)

from timm.models.vision_transformer import Attention
class FaceModel(torch.nn.Module):
    """
    Attributes
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self,num_classes,multi_head=False):
        """Init face model by backbone factorcy and head factory.
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.multi_head= multi_head
        # self.backbone = Xception(0)
        self.backbone = create_RepVGG_B0(num_classes=0)
        self.head = torch.nn.Linear(1280, num_classes)
        # self.head =  torch.nn.Linear(2048,num_classes)
        torch.nn.init.xavier_uniform_(self.head.weight)
        if self.multi_head:
            # self.head_binary = torch.nn.Linear(2048, 2)
            self.head_binary = torch.nn.Linear(1280, 2)
            torch.nn.init.xavier_uniform_(self.head_binary.weight)
    def forward(self, data):
        feat = self.backbone(data)
        pred = self.head.forward(feat)
        if self.multi_head:
            output = self.head_binary(feat)
            return pred,output
        return pred,None
class FaceModel_loc(torch.nn.Module):
    """
    Attributes
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self,num_classes, multi_head=True):
        """Init face model by backbone factorcy and head factory.
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel_loc, self).__init__()
        self.multi_head= multi_head
        self.backbone = create_RepVGG_B0(num_classes=0)
        self.max_pool = torch.nn.MaxPool2d(3,stride=2,padding=1)
        # self.conv1x1 = torch.nn.Conv1d(5,1,3,padding=1)
        # self.in1 = torch.nn.InstanceNorm1d(1728)
        self.IN = torch.nn.ModuleList([torch.nn.InstanceNorm1d(64),torch.nn.InstanceNorm1d(128),torch.nn.InstanceNorm1d(256),torch.nn.InstanceNorm1d(1280)])

        # self.bn1 = torch.nn.BatchNorm1d(1)
        # self.bn2 = torch.nn.BatchNorm1d(512)
        # self.relu = torch.nn.ReLU(inplace=True)
        # self.neck = torch.nn.Linear(1728,512)
        # self.attn = Attention(1728, num_heads=8,qkv_bias=True)
        # self.mask_gap = torch.nn.AdaptiveMaxPool1d(1)
        self.head = torch.nn.Linear(1280, num_classes)
        torch.nn.init.xavier_uniform_(self.head.weight)
        if self.multi_head:
            self.head_binary = torch.nn.Linear(1280,2)
            torch.nn.init.xavier_uniform_(self.head_binary.weight)
            self.head_loc_binary = torch.nn.Linear(1728,2)
            torch.nn.init.xavier_uniform_(self.head_loc_binary.weight)
    def forward(self, data, mask=None):
        output = None
        loc_output = None
        if self.training:
            feat = self.backbone.forward_feature(data)
            feat_whole = feat[-1]
            feats = []
            mask = self.max_pool(mask)
            for l,f in enumerate(feat[1:-1],1):
                mask = self.max_pool(mask) # 4,4,112,112
                fm1 = (mask.flatten(2) @ f.flatten(2).transpose(-2,-1))
                fm1 = self.IN[l](fm1)
                feats.append(fm1)  # /(mask.sum([2,3]).unsqueeze(-1)+1e-8)

            feats = torch.cat(feats,dim=-1) # B,M,C+ ->
            feat_loc = feats.mean(dim=1)
            # feats = self.attn(self.in1(feats))
            #     print(feats)
            # feats = self.mask_gap(feats.transpose(-2,-1))
            # for m in range(feats.shape[1]):
            #     print(f"LN ,feature map max {feats[:, m, :].max()}，mean {feats[:, m, :].mean()}")
            # print(f'Last feature map MAX is {feat[-1].max()},MIN is {feat[-1].min()}')
            # print(f'Before conv1x1 feature map MAX is {feats.max()},MIN is {feats.min()}')
            #feats =self.relu(self.bn1(self.conv1x1(feats)))# .permute(0,2,1) #B,C,N,类似于gap，不过这种方式是在channel上做
            # fuse_mask = self.conv1x1(feats)
            # print(f"fuse_mask ,feature map max {fuse_mask[:, 0, :].max()}，mean {fuse_mask[:, 0, :].mean()}")
            # fuse_mask = self.bn1(fuse_mask)
            # print(f"fuse_mask bn ,feature map max {fuse_mask[:, 0, :].max()}，mean {fuse_mask[:, 0, :].mean()}")
            # print(f"bn ,running mean:{self.bn1.running_mean},running var:{self.bn1.running_var}")
            # feats = self.relu(fuse_mask)# B,M,C+ -> B*1*C
            # feat_loc = feats.flatten(1)
            # print(f'After conv1x1 feature map MAX is {feats.max()},MIN is {feats.min()}')
            # if not math.isfinite(torch.sum(feat)):
            #     print(f"Cat Feat is Nan , stopping training")
            #     sys.exit(1)
            # feat = self.neck(feat)
            # print(f'After neck feature map MAX is {feat.max()},MIN is {feat.min()}')
            # if not math.isfinite(torch.sum(feat)):
            #     print(f"Neck Feat is Nan , stopping training")
            #     sys.exit(1)
            pred = self.head(feat_whole)
            if self.multi_head:
                output= self.head_binary(feat_whole)
                loc_output = self.head_loc_binary(feat_loc)
        else:
            feat_whole = self.backbone(data)
            pred = self.head(feat_whole)
            if self.multi_head:
                output = self.head_binary(feat_whole)
        return pred,output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight',default='',type=str)
    parser.add_argument('--th',type=float)
    parser.add_argument('--gpus',default='')
    parser.add_argument('--multi-head',action='store_true',default=True)
    parser.add_argument('--loc', action='store_true')
    parser.add_argument('--size',default=224,type=int)
    parser.add_argument('--num_classes', default=40, type=int)
    parser.add_argument('--txt', default=None, type=str)
    parser.add_argument('--save_img', action='store_true')
    args = parser.parse_args()
    weight = args.weight
    size = args.size
    th = args.th
    txt = args.txt
    save_img = args.save_img
    assert  weight != "" or th != None
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('Building Model')
    if args.loc:
        model = FaceModel_loc(args.num_classes, args.multi_head)
    else:
        model = FaceModel(args.num_classes,args.multi_head)
    ckpt = torch.load(weight,map_location='cpu')
    new_ckpt=dict()
    for key,value in ckpt['model'].items():
        if 'backbone.' in key or 'head.'  in key or 'head_binary.' in key:
            new_ckpt[key] = value

    model.load_state_dict(new_ckpt, strict=True)
    model.eval()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model = model.to(device)
    transform_test = get_transform('test',size)
    if txt == None:
        abs_path = '/mnt/mfs2/ailun.li/positive_qita/'
        for path in ['zhaji_zhengchang','ZY_wuju_baidu','ZY_wuju_KS','pingan_pos','dandong', 'w_test']: #['w_test','zhaji_zhengchang','ZY_wuju_baidu','ZY_wuju_KS','pingan_pos'
            # print('Building Datasets: ',path)
            images_list = glob.glob(os.path.join(abs_path+path,'*'))
            imgs = []
            for img_path in images_list:
                imgs.append([img_path,0])
            val_dataset = CustomDataset(None,'test',transform_test)
            val_dataset.imgs= imgs
            batch_size = 256 if size == 224 else 128
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    num_workers=16,
                                    shuffle=False,
                                    drop_last=False,
                                    pin_memory=True
                                    )
            total_nums = 0
            FN = 0
            bad_case = []
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc='Evaluate'):
                    images, labels = images.to(device), labels.to(device)
                    bs = images.shape[0]
                    preds, output = model(images)
                    probs = torch.softmax(output, dim=1)
                    score = probs[:, 1]
                    FN += (score > th).sum().item()
                    if len(images[score > th]) > 0:
                        bad_case.append(images[score > th])
                    total_nums += bs
            print(f'{path} FNR@: {FN / total_nums * 100:.4f}')
            # with torch.no_grad():
            #     for img_path in tqdm(images_list):
            #         img = Image.open(img_path)
            #         input = transform_test(img).to(device)
            #         input = input.unsqueeze(0)
            #         output, binary_output = model(input)
            #         if args.multi_head:
            #             prob = torch.softmax(binary_output,dim=1)
            #         else:
            #             real_dist = output[:, :1]
            #             fake_dist = output[:, 1:]
            #             real_dist = torch.mean(real_dist, dim=1, keepdim=True)
            #             fake_dist, _ = torch.max(fake_dist, dim=1, keepdim=True)
            #             dist = torch.cat((real_dist, fake_dist), dim=1)
            #             prob = torch.softmax(dist * model.head.scale, dim=1)
            #         score = prob[0][1]
            #         if score > th:
            #             name = os.path.basename(img_path)
            #             # img.save(os.path.join(output_dir,name.replace('jpg','png')))
            #             # print(name,prob)
            #             neg.append(img_path)
            #         else:
            #             pos.append(img_path)
            # print(f'FRR: {len(neg)/total_nums:.4f}')
    elif os.path.isfile(txt):
        images_labels = []
        pos = []
        neg = []
        val_dataset = CustomDataset(txt, type='test', transforms=transform_test)
        imgs = []
        with open(txt, 'r') as f:
            for line in f.readlines():
                strs = line.strip().split('<blank>')
                if int(strs[1]) == 0:
                    continue
                imgs.append((strs[0], int(strs[1])))
        val_dataset.imgs = imgs
        batch_size = 256 if size == 224 else 128
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=16,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True
                                )
        total_nums = 0
        TN = 0
        acc = 0
        bad_case = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Evaluate'):
                images, labels = images.to(device), labels.to(device)
                bs = images.shape[0]
                preds, output = model(images)
                preds = preds.argmax(1)
                probs = torch.softmax(output, dim=1)
                score = probs[:, 1]
                TN += (score > th).sum().item()
                if len(images[score < th]) >0:
                    bad_case.append(images[score < th])
                preds = preds[score > th]
                labels = labels[score > th]
                total_nums += bs
                acc += preds.eq(labels).sum().item()
        if save_img:
            output_dir = os.path.join('/'.join(weight.split('/')[:-1]), 'bad_case')
            os.makedirs(output_dir, exist_ok=True)
            bad_case = torch.cat(bad_case)
            for i,img in enumerate(bad_case):
                name =f'bad_{i}.png'
                imgs = img.permute(1,2,0).cpu().numpy()
                mean = np.array([0.560, 0.449, 0.407])
                std=np.array([0.248, 0.229, 0.223])
                imgs = imgs*std+mean
                cv2.imwrite(os.path.join(output_dir,name),imgs[:,:,::-1]*255.)
        print(f'acc@multi: {acc / total_nums * 100:.4f}')
        print(f'TNR@: {TN / total_nums * 100:.4f}')
    elif os.path.isdir(txt):
        if '实网' in txt:
            abs_path = '/mnt/mfs2/ailun.li/实网黑产/'
            for path in ["fdd_21_1"  ,"hz_22_注入" , "nx_21_1" , "tx_22_1" , "zy_21_1" , "zy_22_1"]:  #
                # print('Building Datasets: ',path)
                images_list = glob.glob(os.path.join(abs_path + path, '*'))
                imgs = []
                for img_path in images_list:
                    imgs.append([img_path, 1])
                val_dataset = CustomDataset(None, 'test', transform_test)
                val_dataset.imgs = imgs
                batch_size = 256 if size == 224 else 128
                val_loader = DataLoader(val_dataset,
                                        batch_size=batch_size,
                                        num_workers=16,
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True
                                        )
                total_nums = 0
                TN = 0
                bad_case = []
                with torch.no_grad():
                    for images, labels in tqdm(val_loader, desc='Evaluate'):
                        images, labels = images.to(device), labels.to(device)
                        bs = images.shape[0]
                        preds, output = model(images)
                        probs = torch.softmax(output, dim=1)
                        score = probs[:, 1]
                        TN += (score > th).sum().item()
                        if len(images[score < th]) > 0:
                            bad_case.append(images[score > th])
                        total_nums += bs
                print(f'{path}, data: {len(val_dataset)} TNR@: {TN / total_nums * 100:.4f}')
        else:
            images_list = glob.glob(os.path.join(txt, '*'))
            imgs = []
            for img_path in images_list:
                imgs.append([img_path, 1])
            val_dataset = CustomDataset(None, 'test', transform_test)
            val_dataset.imgs = imgs
            batch_size = 256 if size == 224 else 128
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    num_workers=16,
                                    shuffle=False,
                                    drop_last=False,
                                    pin_memory=True
                                    )
            total_nums = 0
            TN = 0
            bad_case = []
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc='Evaluate'):
                    images, labels = images.to(device), labels.to(device)
                    bs = images.shape[0]
                    preds, output = model(images)
                    probs = torch.softmax(output, dim=1)
                    # print(probs)
                    score = probs[:, 1]
                    TN += (score > th).sum().item()
                    if len(images[score < th]) > 0:
                        bad_case.append(images[score > th])
                    total_nums += bs
            print(f'{txt}, data: {len(val_dataset)} TNR@: {TN / total_nums * 100:.4f}')
            #






"""
python result_analysis.py  --gpus 1  --num_classes 40 --weight run/Jul_5_RepVGG_MultiSoftmax_Huokai_Dg_ClsW91_weight1.0_CentesLoss1.0_size224/epoch_45.pth  --th 0.991327 
python result_analysis.py  --gpus 6  --num_classes 40 --weight run/Jul_14_RepVGG_MultiSoftmax_HuanlianW_Dg_ClsW81_weight1.0_size224/epoch_39.pth --th  0.00135840673 --txt 
python result_analysis.py  --gpus 1  --num_classes 24 --weight run/Jul_11_Xception_MultiSoftmax_CrazytalkNewClean_Dg_ClsW13_weight1.0_size224/epoch_49.pth  --th 0.02144457958638668 --txt data_txts/8g8/huokai_test_PIL_resize224_crazytalknew_clean.txt
"""

