import torch
import os
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from repvgg.repvgg_hy import RepVGG, create_RepVGG_B0, repvgg_model_convert
from torchvision import transforms
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


class FaceModel(torch.nn.Module):
    """
    Attributes
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, num_classes, multi_head=True):
        """Init face model by backbone factorcy and head factory.
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.multi_head = multi_head
        # self.backbone = Xception(0)
        self.backbone = create_RepVGG_B0(num_classes=0)
        self.head = torch.nn.Linear(1280, 5, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.head =  torch.nn.Linear(2048,num_classes)
        torch.nn.init.xavier_uniform_(self.head.weight)
        if self.multi_head:
            # self.head_binary = torch.nn.Linear(2048, 2)
            self.head_binary = torch.nn.Linear(1280, 1, bias=True)
            torch.nn.init.xavier_uniform_(self.head_binary.weight)

    def forward(self, data):
        B = data.shape[0]
        feat = self.backbone.forward(data)
        feat = self.gap(feat).view(B, -1)
        multi_output = self.head(feat)
        output = self.head_binary(feat)
        return multi_output, output


class Demo():
    def __init__(self, model_path, img_size=224):
        self.model = FaceModel(num_classes=2)
        ckpt = torch.load(model_path)
        # print(ckpt.keys())
        # exit()
        self.model.load_state_dict(ckpt['model'], strict=False)

        print("convert repvgg model to deploy")
        self.model.backbone = repvgg_model_convert(self.model.backbone)

        self.model.cuda()
        self.model.eval()

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.560, 0.449, 0.407], std=[0.248, 0.229, 0.223])
        ])
        self.img_size = img_size

    def predict(self, img_path):
        # img = Image.open(img_path).convert('RGB')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transforms(img)
        img = torch.unsqueeze(img, 0).cuda()
        with torch.no_grad():
            multi_outputs, outputs = self.model(img)
            # print(outputs)
        multi_output = torch.sigmoid(multi_outputs).tolist()[0]
        output = torch.sigmoid(outputs).item()

        return img_path, multi_output, output


if __name__ == '__main__':

    demo = Demo('/home/ailun.li/Multi_softmax/run/198_质量模块_224/best.pth')

    _list = []
    data_list = '/home/ailun.li/Multi_softmax/data_txts/198_质量模块/quality_all_test.txt'
    with open(data_list, 'rt') as file:
        for line in file.readlines():
            parts = line.strip().split('<blank>')
            if len(parts) >= 2:
                image_path = parts[0]
                binary_label = int(parts[1])
                multi_label = parts[2]
                _list.append((image_path, binary_label, multi_label))

    print("all file:", len(_list))

    with open('质量_pos_mul.txt', 'w') as pos_file, open('质量_neg_mul.txt', 'w') as neg_file:
        # with open('tmp.txt', 'w') as log_file:
        for img_path, y, multi_y in tqdm(_list):
            try:
                _, res_mul, res = demo.predict(img_path)
                if y == 0:
                    pos_file.write(f'{img_path},{res},{y},{res_mul},{multi_y}\n')
                else:
                    neg_file.write(f'{img_path},{res},{y},{res_mul},{multi_y}\n')
            except:
                continue
