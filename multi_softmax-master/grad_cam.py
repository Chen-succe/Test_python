# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:37

@author: mick.yi

"""
import numpy as np
import cv2
from util.datasets import  CustomDataset
from torchvision import transforms
from repvgg.repvgg import create_RepVGG_B0
from tqdm import tqdm
import random
from backbone.backbone_def import BackboneFactory
from head.head_def import  HeadFactory
from models import *
from main import FaceModel
from PIL import Image
import shutil
import glob
from timm.models.vision_transformer import Attention

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index=None,is_binary=True):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        outputs, binary_outputs = self.net(inputs, index if index else 0)  # [1,num_classes]
        if is_binary:
            if index is None:
                index = np.argmax(binary_outputs.cpu().data.numpy())
            else:
                index = 1 if index > 0 else 0
            target = binary_outputs[0][index]
        # real_dist = output[:, :1]
        # fake_dist = output[:, 1:]
        # real_dist = torch.mean(real_dist, dim=1, keepdim=True)
        # fake_dist, _ = torch.max(fake_dist, dim=1, keepdim=True)
        # output = torch.cat((real_dist, fake_dist), dim=1)
        else:
            if index is None:
                index = np.argmax(outputs.cpu().data.numpy())
            target = outputs[0][index]
        # print('output',outputs)
        # print('binary_outputs', binary_outputs)
        print(f'label:{index},Fake Prob:',torch.softmax(binary_outputs,dim=1)[0][1])
        target.backward()
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]
        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]
        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # cam = feature
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU
        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (size, size))
        return cam

class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index=None,is_binary=True):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        outputs, binary_outputs = self.net(inputs, index if index else 0)  # [1,num_classes]
        if is_binary:
            if index is None:
                index = np.argmax(binary_outputs.cpu().data.numpy())
            else:
                index = 1 if index>0 else 0
            target = binary_outputs[0][index]
        # real_dist = output[:, :1]
        # fake_dist = output[:, 1:]
        # real_dist = torch.mean(real_dist, dim=1, keepdim=True)
        # fake_dist, _ = torch.max(fake_dist, dim=1, keepdim=True)
        # output = torch.cat((real_dist, fake_dist), dim=1)
        else:
            if index is None:
                index = np.argmax(outputs.cpu().data.numpy())
            target = outputs[0][index]
        target.backward()
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (size, size))
        return cam


# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:45

@author: mick.yi

"""
import torch
from torch import nn
import numpy as np


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """

        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=None,is_binary=True):
        """

        :param inputs: [1,3,H,W]
        :param index: class_id
        :return:
        """
        self.net.zero_grad()
        outputs, binary_outputs = self.net(inputs, index if index else 0)  # [1,num_classes]
        if is_binary:
            if index is None:
                index = np.argmax(binary_outputs.cpu().data.numpy())
            else:
                index = 1 if index > 0 else 0
            target = binary_outputs[0][index]
        # real_dist = output[:, :1]
        # fake_dist = output[:, 1:]
        # real_dist = torch.mean(real_dist, dim=1, keepdim=True)
        # fake_dist, _ = torch.max(fake_dist, dim=1, keepdim=True)
        # output = torch.cat((real_dist, fake_dist), dim=1)
        else:
            if index is None:
                index = np.argmax(outputs.cpu().data.numpy())
            target = outputs[0][index]
        target.backward()

        return inputs.grad[0]  # [3,H,W]

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

def prepare_input(image):
    image = image.copy()
    # 归一化
    # means = np.array([0.485, 0.456, 0.406])
    # stds = np.array([0.229, 0.224, 0.225])
    means = np.array([0.560, 0.449, 0.407])
    stds = np.array([0.248, 0.229, 0.223])

    image -= means
    image /= stds
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维
    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.cpu().data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb
import os
from skimage import io

def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    os.makedirs(output_dir,exist_ok=True)
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

class FaceModel_loc(torch.nn.Module):
    """
    Attributes
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self,num_classes, multi_head=False):
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
        # self.in1 = torch.nn.LayerNorm(1728)
        # self.in1 = torch.nn.InstanceNorm1d(1728)
        self.IN = nn.ModuleList([torch.nn.InstanceNorm1d(64),torch.nn.InstanceNorm1d(128),torch.nn.InstanceNorm1d(256),torch.nn.InstanceNorm1d(1280)])

        # self.bn1 = torch.nn.BatchNorm1d(1)
        # self.bn2 = torch.nn.BatchNorm1d(512)
        # self.relu = torch.nn.ReLU(inplace=True)
        # self.neck = torch.nn.Linear(1728,512)
        # self.attn = Attention(1728, num_heads=8,qkv_bias=True)
        # self.mask_gap = nn.AdaptiveMaxPool1d(1)
        self.head = torch.nn.Linear(1280, num_classes)
        torch.nn.init.xavier_uniform_(self.head.weight)
        if self.multi_head:
            self.head_binary = torch.nn.Linear(1280,2)
            torch.nn.init.xavier_uniform_(self.head_binary.weight)
            self.head_loc_binary = torch.nn.Linear(1728,2)
            torch.nn.init.xavier_uniform_(self.head_loc_binary.weight)
    def forward(self, data, mask):
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
                # for m in range(fm1.shape[1]):
                #     print(f"stage {l},mask{m},feature map max {fm1[:,m,:].max()}，mean {fm1[:,m,:].mean()}")
                feats.append(fm1) # /(mask.sum([2,3]).unsqueeze(-1)+1e-8)
            feats = torch.cat(feats,dim=-1) # B,M,C+ ->
            if feats.isinf().sum()+feats.isnan().sum() > 0:
                print('cat',feats.max(),feats.min())

            # for m in range(feats.shape[1]):
            #     print(f"Cat ,feature map max {feats[:, m, :].max()}，mean {feats[:, m, :].mean()}")
            # feats = self.neck(self.in1(feats))
            # if feats.isinf().sum()+feats.isnan().sum() > 0:
            #     print('neck',feats.max(),feats.min())
            #     print(feats)
            feats = self.attn(self.in1(feats))
            if feats.isinf().sum()+feats.isnan().sum() > 0:
                print('atten',feats.max(),feats.min())
                print(feats)
            feats = self.mask_gap(feats.transpose(-2,-1))
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
            feat_loc = feats.flatten(1)
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

class FaceModel(torch.nn.Module):
    """
    Attributes
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self,num_classes,multi_head=True):
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
    def forward(self, data,label=None):
        feat = self.backbone(data)
        pred = self.head.forward(feat)
        if self.multi_head:
            output = self.head_binary(feat)
            return pred,output
        return pred,None


import argparse
from util.datasets import get_transform
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight',default=None,type=str)
    parser.add_argument('--th',type=float)
    parser.add_argument('--gpus',default='')
    parser.add_argument('--multi-head',action='store_true',default=True)
    parser.add_argument('--loc', action='store_true')
    parser.add_argument('--size',default=224,type=int)
    parser.add_argument('--num_classes', default=24, type=int)
    parser.add_argument('--txt', default='data_txts/8g8/huokai_test_PIL_resize224_crazytalknew.txt', type=str)
    parser.add_argument('--index', default=1, type=int)
    # test_list = '/mnt/mfs2/ailun.li/positive_qita/zhaji_zhengchang'
    #['w_test','zhaji_zhengchang','ZY_wuju_baidu','ZY_wuju_KS','pingan_pos']
    args = parser.parse_args()
    weight = args.weight
    txt =  args.txt
    size = args.size
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    assert weight!=None and txt!=None
    images_labels = []
    output_dir = '/'.join(weight.split('/')[:-1])
    if '黑产' not in txt:
        if os.path.isfile(txt):
            with open(txt, 'r') as f:
                for k, l in enumerate(tqdm(f.readlines())):
                    strs = l.strip().split('<blank>')
                    if int(strs[1])==0:
                        continue
                    images_labels.append((strs[0],int(strs[1])))
        elif os.path.isdir(txt):
            images_list = glob.glob(os.path.join(txt, '*'))
            for img in images_list:
                images_labels.append((img, 0))
        output_dir = os.path.join(output_dir, 'grad_cam_badcase' if 'bad_case' in txt else 'grad_cam')

    else:
        images_list = glob.glob(os.path.join(txt, '*/*.png'))+glob.glob(os.path.join(txt, '*/*.jpg') ) +glob.glob(os.path.join(txt, '*/*.jpeg'))
        for img in images_list:
            images_labels.append((img, 1))
        output_dir = os.path.join(output_dir,'grad_cam_sw')
    random.shuffle(images_labels)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if args.loc:
        model = FaceModel_loc(num_classes=args.num_classes,multi_head=args.multi_head)
    else:
        model = FaceModel(num_classes=args.num_classes,multi_head=args.multi_head)
    checkpoint = torch.load(weight, map_location='cpu')
    new_ckpt=dict()
    for key,value in checkpoint['model'].items():
        if 'backbone.' in key or 'head.'  in key or 'head_binary.' in key:
            new_ckpt[key] = value
    model.load_state_dict(new_ckpt,strict=True)
    model.eval()
    model.to(device)
    # random.shuffle(images_labels)
    # images_labels.sort()
    # model = model.cuda()
    # model = model.cuda()
    transform_test = get_transform('test',size)
    index = args.index
    if index==0:
        output_dir=os.path.join(output_dir,'attention_real')
    else:
        output_dir=os.path.join(output_dir, 'attention_fake')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i,(path,label) in enumerate(tqdm(images_labels)):
        print('    **********************    ')
        print(path)
        print('Label : ', label)
        img = Image.open(path)
        img = img.resize((size,size))
        # img = io.imread(path)
        # img = np.float32(cv2.resize(img, (size, size))) / 255
        # img = np.float32(img.resize((224, 224),Image.BILINEAR)) / 255
        # inputs = prepare_input(img)
        inputs = transform_test(img)
        inputs = inputs.unsqueeze(0).to(device)
        inputs.requires_grad = True
        img = np.float32(np.array(img)) / 255.
        image_dict = {}
        layer_name = get_last_conv_name(model)
        grad_cam = GradCAM(model, layer_name)


        mask = grad_cam(inputs,index=index,is_binary=True)  # cam mask
        image_dict['img'] = np.uint8(img*255)
        image_dict['cam'], _ = gen_cam(img, mask)
        grad_cam.remove_handlers()
        # Grad-CAM++
        grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
        mask_plus_plus = grad_cam_plus_plus(inputs,index=index,is_binary=True)  # cam mask
        image_dict['cam++'], _ = gen_cam(img, mask_plus_plus)
        grad_cam_plus_plus.remove_handlers()
        # GuidedBackPropagation
        gbp = GuidedBackPropagation(model)
        inputs.grad.zero_()  # 梯度置零
        grad = gbp(inputs,index=index,is_binary=True)
        gb = gen_gb(grad)
        image_dict['gb'] = norm_image(gb)
        # 生成Guided Grad-CAM
        cam_gb = gb * mask[..., np.newaxis]
        image_dict['cam_gb'] = norm_image(cam_gb)
        name = os.path.basename(path)
        save_image(image_dict, os.path.basename(path),'best',output_dir) # 'grad_cam_visual'
        if i ==20 and '实网' not in txt:
            break

"""
python grad_cam.py  --gpus 4 --num_classes 24 --weight run/Jul_11_Xception_MultiSoftmax_CrazytalkNewClean_Dg_ClsW13_weight1.0_size224/epoch_49.pth  --txt data_txts/8g8/huokai_test_PIL_resize224_crazytalknew_clean.txt   
python grad_cam.py  --gpus 6 --num_classes 24 --weight run/Jul_14_RepVGG_MultiSoftmax_HuanlianW_Dg_ClsW81_weight1.0_size224/epoch_39.pth --txt data_txts/8g8/huokai_test_PIL_resize224_crazytalknew_clean.txt  
"""

