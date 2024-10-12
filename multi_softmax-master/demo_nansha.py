import torch
import os
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from repvgg import RepVGG, create_RepVGG_B0, repvgg_model_convert
from torchvision import transforms
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
        self.backbone = create_RepVGG_B0(num_classes=0, use_checkpoint=False)
        self.head = torch.nn.Linear(1280, num_classes)
        # torch.nn.init.xavier_uniform_(self.head.weight)
        if self.multi_head:
            self.head_binary = torch.nn.Linear(1280, 2)
            # torch.nn.init.xavier_uniform_(self.head_binary.weight)

    def forward(self, data):
        feat = self.backbone.forward(data)
        # feat = F.dropout(feat,training=self.training)
        output = self.head_binary(feat)
        return output
        # pred = self.head(feat)
        # if self.multi_head:
        #    output = self.head_binary(feat)
        #    return pred,output
        # return pred,None


class Demo():
    def __init__(self, model_path, img_size=224):
        self.model = FaceModel(num_classes=41)
        ckpt = torch.load(model_path)
        # print(ckpt.keys())
        # exit()
        self.model.load_state_dict(ckpt['model'])

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
            outputs = self.model(img)
        output = nn.Softmax(dim=1)(outputs).tolist()[0][1]
        return img_path, output


if __name__ == '__main__':

    demo = Demo('run/Nov_3_RepVGG_MultiSoftmax_all_weight1.0_size224/epoch_8.pth')

    _list = []
    data_list = 'data_list/tmp.txt'
    s = open(data_list, 'rt')
    for line in s.readlines():
        _list.append((line.strip().split('<blank>')[0], int(line.strip().split('<blank>')[1])))

    print("all file:", len(_list))

    with open('t.txt', 'w') as pos_file, open('res_nansha_neg_all_test.txt', 'w') as neg_file:
        # with open('tmp.txt', 'w') as log_file:
        for img_path, y in tqdm(_list):
            _, res = demo.predict(img_path)
            if y == 0:
                pos_file.write(f'{img_path},{res},{y}\n')
            else:
                neg_file.write(f'{img_path},{res},{y}\n')
