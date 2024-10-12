import torch
from timm.models import create_model
from models import *
from util import transform_test
import os
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import time
from torchvision.models import resnet18
from convnextv2 import convnextv2_atto, convnextv2_femto, convnext_pico, convnextv2_nano
from repvgg import create_RepVGG_A0, create_RepVGG_A1, create_RepVGG_A2, create_RepVGG_B0


class Demo():
    def __init__(self, model_path, img_size=224):
        self.model = create_model(
            'efficientformerv2_s2',
            num_classes=2,
            distillation=False,
            pretrained=None
        )
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()

        self.transforms = transform_test
        self.img_size = img_size

    def predict(self, img_path):
        t0 = time.time()
        img = Image.open(img_path).convert('RGB')
        t1 = time.time()
        img = self.transforms(img)
        t2 = time.time()
        img = torch.unsqueeze(img, 0).cuda()
        t3 = time.time()

        with torch.no_grad():
            preds = self.model(img)
        output = nn.Softmax(dim=1)(preds).tolist()[0][1]
        t4 = time.time()
        return t1-t0, t2-t1, t3-t2, t4-t3, output


if __name__ == '__main__':

    demo = Demo('weight_0221/epoch_95_98.9647.pth')

    # root_dirs = [
    #         ('/mnt/mfs2/zhao.xie/data/huanlian_test_crop/', 1),
    #         ]
    root_dirs = [
           ('/mnt/mfs2/ailun.li/positive_new/p_test_crop/', 0),
           ]
    _list = []
    for (root_dir, y) in root_dirs:
        for _root, _dirs, _files in os.walk(root_dir):
            for _file in _files:
                if _file.endswith('.jpg') or _file.endswith('.png') or _file.endswith('.jpeg'):
                    img_path = os.path.join(_root, _file)
                    _list.append((img_path, y))
        print(root_dir, len(_list))
    print("all file:", len(_list))

    with open('swapface_eformer_0220_ep95_acc9896_pos_time.csv', 'w') as log_file:
        for img_path, y in tqdm(_list):
            #if 'mask.png' in img_path:
            #    continue
            t_start = time.time()
            t01, t12, t23, t34, res = demo.predict(img_path)
            t_end = time.time()
            log_file.write(f'{img_path},{t01},{t12},{t23},{t34},{res},{y}\n')

