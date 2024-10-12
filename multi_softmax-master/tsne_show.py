import argparse
import datetime
import os
import shutil
import time
import json
from pathlib import Path
from repvgg import create_RepVGG_B0
from models import *
from util import *
from torch.nn.parallel import DataParallel as DP
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torch
import logging
from matplotlib import pyplot as plt
# import seaborn as sns
from sklearn.manifold import TSNE
from timm.utils.metrics import AverageMeter
# from backbone.RepVGG import create_RepVGG_B0
from util import BatchMMD_loss
def plot_embedding_2D(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # fig = plt.figure()
    fig = plt.figure()
    plt.axis('off')
    colors =['green','red','purple','orange','yellow','blue','cyan','gray','chocolate','gold','hotpink','slateblue','brown','DarkTurquoise',
             'MediumAquamarine','Moccasin','DarkSlateGray','lemonchiffon','palegoldenrod','honeydew','cornflowerblue','deeppink','tan','lemonchiffon','black']

    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str("●"),# str(int(label[i])), # 用什么符号绘制当前数据点
                 color= colors[label[i]], # plt.cm.Set1(int(label[i]) / 10),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.show()
    return fig

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
        # self.backbone = Xception(0, pretrained='saved_weight/xception-43020ad28.pth')
        self.backbone = create_RepVGG_B0(num_classes=0)
        self.head = torch.nn.Linear(1280, num_classes)
        # self.head = torch.nn.Linear(2048,num_classes)
        # origin repvgg
        # self.backbone = create_RepVGG_B0()
        # self.head = torch.nn.Linear(512, num_classes,bias=False)
        torch.nn.init.xavier_uniform_(self.head.weight)
        if self.multi_head:
            # self.head_binary = torch.nn.Linear(512, 2,bias=False)
            self.head_binary = torch.nn.Linear(1280, 2)
            # self.head_binary = torch.nn.Linear(2048, 2)
            torch.nn.init.xavier_uniform_(self.head_binary.weight)
    def forward(self, data):
        feat = self.forward_feature(data)
        feat = F.dropout(feat,training=self.training)
        pred = self.head(feat)
        if self.multi_head:
            output = self.head_binary(feat)
            return feat,pred,output
        return feat,pred,None
    def forward_feature(self,data):
        return  self.backbone(data)


def get_args_parser():
    parser = argparse.ArgumentParser(
        'norm cls training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=256, type=int)

    # /mnt/mfs2/ailun.li/data_txt/huokai/huokai_train.txt
    # data_txts/huokai_train_PIL_resize224.txt
    parser.add_argument('--val_list', default='data_txts/8g8/huokai_test_PIL_resize224_positive.txt', type=str,help='dataset path')
    parser.add_argument('--val_add_list', default=[], nargs='+', help='add dataset path')
    # /mnt/mfs2/ailun.li/data_txt/huokai/huokai_test.txt
    # data_txts/huokai_test_PIL_resize224.txt
    # 'data_txts/8g8/huokai_test_PIL_resize224_crazytalknew_clean.txt'
    # Model parameters
    parser.add_argument("--head_conf_file", type = str, default='./head/head_conf.yaml', help = "the path of head_conf.yaml.")
    parser.add_argument("--multi_head", action='store_true', default=True,help="")
    parser.add_argument('--input-size', default=224,type=int, help='images input size')
    parser.add_argument('--num_classes', default=24,type=int, help='num classes')
    # * Finetuning params
    parser.add_argument('--output_dir', default='./weight',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--gpus', default='0,1,2,3', type=str,help='num gpus')
    parser.add_argument('--ema', action='store_true', help='Model ema')
    parser.add_argument('--tsne', action='store_true', help='feature tsne')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--finetune', default=None, help='checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--eval', action='store_true',help='Perform evaluation only')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin-mem', action='store_true',help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    return parser
def fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main(args):
    fix_random(args.seed)
    dataset_val = CustomDataset(args.val_list, type='test', transforms=get_transform('test',resize=args.input_size))
    print(args.val_add_list)
    for tal in args.val_add_list:
        dataset_val_add = CustomDataset(tal, type='test', transforms=None)
        dataset_val.imgs+=dataset_val_add.imgs
    print('val data:', len(dataset_val))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )
    model = FaceModel(num_classes=args.num_classes,multi_head=args.multi_head)
    # model = create_RepVGG_B0(num_classes=args.num_classes)
    # model = torchvision.models.resnet34(False,num_classes=args.num_classes)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model)
        logging.info(f"load pretrain :{args.finetune}")
    model.to(args.device)
    model = DP(model)
    # model_without_dp = model.module
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    model.eval()
    BMMD = BatchMMD_loss()
    BMMD_avg = AverageMeter()
    features = None
    labels_multi = None
    labels_binary = None
    pos_list, neg_list = [], []
    pos_idxOffeature = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader_val):
            images = images.to(args.device, non_blocking=True)
            # targets = targets.to(args.device, non_blocking=True)
            feature,outputs, binary_outputs = model(images)
            bi_labels = targets.clone()
            bi_labels[bi_labels < 1] = 0
            bi_labels[bi_labels >= 1] = 1
            probs = F.softmax(binary_outputs, dim=1)
            scores = probs[:, 1].tolist()
            bmmd_loss = BMMD(feature,targets.clone().cuda())
            BMMD_avg.update(bmmd_loss.item(),len(targets))
            for i in range(len(targets)):
                if targets[i] == 0:
                    pos_list.append(scores[i])
                    pos_idxOffeature.append(i + (len(features) if features is not None else 0))
                else:
                    neg_list.append(scores[i])
            feature = feature.detach().clone().cpu()
            if features is None:
                features = feature
                labels_multi = targets.clone()
                labels_binary = bi_labels.clone()
            else:
                features = torch.cat([features, feature])
                labels_multi = torch.cat([labels_multi, targets.clone()])
                labels_binary = torch.cat([labels_binary, bi_labels.clone()])
    print("MDDS",BMMD_avg.avg)
    result_2D = tsne_2D.fit_transform(features)
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)) # .sort()[0]
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)) # .sort()[0]
    print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')
    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        values,indices = all_pos_list.topk(threshold)
        thresh = values[-1]
        idx = indices[-1]
        feature_idx = pos_idxOffeature[idx]
        labels_m = labels_multi.clone()
        labels_b = labels_binary.clone()
        result = deepcopy(result_2D)
        result[-1],result[feature_idx] = result_2D[feature_idx],result_2D[-1]
        labels_m[feature_idx] = labels_m[-1]
        labels_b[feature_idx] = labels_b[-1]
        labels_m[-1] = 24
        labels_b[-1] = 24
        fig_multi = plot_embedding_2D(result, labels_m)
        fig_binary = plot_embedding_2D(result, labels_b)
        fig_multi.savefig(os.path.join(args.output_dir, f'tsne_multi{threshold_r}.jpg'))
        fig_binary.savefig(os.path.join(args.output_dir, f'tsne_binary{threshold_r}.jpg'))
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
from copy import deepcopy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'EfficientFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.finetune:
        args.output_dir = 'run/' + args.finetune.strip().split('/')[1]
    main(args)
"""
python tsne_show.py --gpus 5 --finetune run/Jul_5_RepVGG_MultiSoftmax_Huokai_Dg_ClsW91_weight1.0_MaxBatchMMD1.0_margin2.0_size224/epoch_47.pth  --val_list  data_txts/8g8/huokai_test_PIL_resize224_8g8.txt --batch-size 384
"""