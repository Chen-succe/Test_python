import argparse
import datetime
import os
import shutil
import time
import json
from pathlib import Path


from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from repvgg import create_RepVGG_B0
from models import *
from util import *
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss
from amsoftmax import AMSoftmax
import random
import numpy as np
import torch
import logging
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory

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
        self.backbone = create_RepVGG_B0(num_classes=0,use_checkpoint='saved_weight/RepVGG-B0-train.pth')
        self.head = torch.nn.Linear(1280, num_classes)
        torch.nn.init.xavier_uniform_(self.head.weight)
        if self.multi_head:
            self.head_binary = torch.nn.Linear(1280, 2)
            torch.nn.init.xavier_uniform_(self.head_binary.weight)
    def forward(self, data):
        feat = self.backbone.forward(data)
        # 使用dropout 对特征进行正则化
        feat = F.dropout(feat,training=self.training)
        pred = self.head(feat)
        if self.multi_head:
            # 如果启用了多头模型，使用head_binary进行额外的任务预测
            output = self.head_binary(feat)
            return pred,output
        return pred,None
def get_args_parser():
    parser = argparse.ArgumentParser(
        'norm cls training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    # Dataset parameters
    parser.add_argument('--train_list', default='/home/ailun.li/Multi_softmax/data_txts/198_多分类标签/all_train_320.txt', type=str,help='dataset path')
    parser.add_argument('--train_add_list', default='/home/ailun.li/Multi_softmax/data_txts/198_多分类标签/w_train.txt', type=str, help='dataset path')
    # /mnt/mfs2/ailun.li/data_txt/all/huanlian/newlabel/huanlian_train_newlabel.txt
    # /mnt/mfs2/ailun.li/data_txt/all/huanlian/newlabel/huanlian_train_newlabel.txt
    # data_txts/all_train_PIL_resize224.txt
    # ./huokai_train_PIL_resize224.txt
    parser.add_argument('--val_list', default='/home/ailun.li/Multi_softmax/data_txts/198_多分类标签/all_val_320.txt', type=str,help='dataset path')
    # /mnt/mfs2/ailun.li/data_txt/all/huanlian/newlabel/huanlian_val_newlabel.txt'
    # data_txts/all_test_PIL_resize224.txt
    parser.add_argument('--test_list', default='/home/ailun.li/Multi_softmax/data_txts/198_多分类标签/all_test_320.txt', type=str, help='dataset path')
    # /mnt/mfs2/ailun.li/data_txt/all/huanlian/newlabel/huanlian_test_newlabel.txt
    # data_txts/all_test_PIL_resize224.txt
    #/mnt/mfs2/ailun.li/data_txt/huokai/huokai_test.txt
    # ./huokai_test_resize224.txt
    # ./huokai_test_PIL_resize224.txt
    # Model parameters
    parser.add_argument("--backbone_type", type = str,default='RepVGG', help = "Mobilefacenets, Resnet.")
    parser.add_argument("--backbone_conf_file", type = str, default='./backbone/backbone_conf.yaml', help = "the path of backbone_conf.yaml.")
    parser.add_argument("--head_type", type = str, default='AM-Softmax',help = "mv-softmax, arcface, npc-face.")
    parser.add_argument("--head_conf_file", type = str, default='./head/head_conf.yaml', help = "the path of head_conf.yaml.")
    parser.add_argument("--multi_head", action='store_true', default=True,help="")
    parser.add_argument('--input-size', default=224,type=int, help='images input size')
    parser.add_argument('--num_classes', default=47,type=int, help='num classes')
    parser.add_argument('--bmmd_margin', default=1.0, type=float, help='margin of bmmd loss')
    parser.add_argument('--weight', default=1.0,type=float, help='softmax weight')
    parser.add_argument('--bi_class_weight', default=[3,1], type=list, help="Binary classify weight")
    parser.add_argument('--consistency_loss_weight', default=1.0, type=float, help="consistency loss weight")
    parser.add_argument('--RDrop', default=0.0, type=float, help="RDrop, Only dropout in Model")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--output_dir', default='./198_cpu_weight',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--gpus', default='0,1,2,3', type=str,help='num gpus')
    parser.add_argument('--ema',action='store_true', help='Model ema')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-6, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    #clip-grad 用于限制梯度的范数，防止梯度值过大导致梯度爆炸的问题。梯度范数是梯度向量中所有元素的平方和的平方根，用于度量梯度的大小。
    #如果梯度的范数超过了指定的值，则会对梯度进行裁剪，将梯度的范数缩放到指定的范围内
    parser.add_argument('--clip-grad', type=float, default=10, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    #clip-mode 用于梯度裁剪的模式，包括：norm,value,agc
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.025,
                        help='weight decay (default: 0.025)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 2e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.5, metavar='PCT',help='Color jitter factor (default: 0.5)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,help='Do not random erase first (clean) augmentation split')

    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--eval', action='store_true',help='Perform evaluation only')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin-mem', action='store_true',help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # DDP
    parser.add_argument('--DDP', action='store_true')
    parser.add_argument('--repeated-aug', action='store_true',default=True)
    parser.add_argument('--sync-bn', action='store_true',default=True,help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist-eval', action='store_true',default=True, help='Enabling distributed evaluation ')
    return parser
def fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
# from util.datasets import  strong_transform_train,identity,transform_test
# import webdataset
# class MyWebdataset(webdataset.WebDataset):
#     def __init__(self,length,urls,resampled=False):
#         super(MyWebdataset,self).__init__(urls,resampled=resampled)
#         self.length = length
#     def __len__(self):
#         return self.length
from tqdm import tqdm
def main(args):
    if args.DDP:
        utils.init_distributed_mode(args)
    fix_random(args.seed)
    dataset_train = CustomDataset(args.train_list, type='train', transforms=get_transform('train',resize=args.input_size))
    add_imgs = []
    with open(args.train_add_list, 'rt') as f:
        for line in f.readlines():
            strs = line.strip().split('<blank>')
            add_imgs.append((strs[0], int(strs[1])))
    dataset_train.imgs += add_imgs
    dataset_val = CustomDataset(args.val_list, type='test', transforms=get_transform('test',resize=args.input_size))
    dataset_test = CustomDataset(args.test_list, type='test', transforms=get_transform('test', resize=args.input_size))
    if args.DDP:
        #分别获取当前分布式训练环境中进程数'num_tasks'和当前进程的全局排名‘global_bank’
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        #根据 args.repeated_aug 和 args.dist_eval 的值，选择合适的数据采样器（Sampler）
        logging.info(f"num tasks : {num_tasks}")
        logging.info(f"global_rank : {global_rank}")
        if args.repeated_aug:
            # 如果args.repeated_aug 为true，则使用自定义的RASampler采样器，它是一个带重复数据增强的采样器，用于数据增广
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            # 如果args.repeated_aug 为Flase，则使用PyTorch 提供的 torch.utils.data.DistributedSampler 采样器，它用于分布式训练中，在每个进程上对数据进行划分和采样。
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            #如果 args.dist_eval 为 True，则使用 torch.utils.data.DistributedSampler 采样器来进行验证数据的分布式采样。这样做是为了确保在分布式环境下验证数据也能按照一定规则进行采样
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            #SequentialSampler 是 PyTorch 提供的一个简单采样器，它按照数据集中的顺序依次对数据进行采样，即从头到尾依次取样本，不进行随机重排
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            worker_init_fn=dataset_train.worker_init_fn,
            drop_last=True,
            shuffle=True
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )

    # backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    # if args.input_size != 224:
    #     backbone_factory.backbone_param['out_w']=args.input_size // 32
    #     backbone_factory.backbone_param['out_h'] = args.input_size // 32
    #
    # head_factory = HeadFactory(args.head_type, args.head_conf_file)
    # logging.info(f'Backbone Parameters: {args.backbone_type}')
    # logging.info(backbone_factory.backbone_param)
    # logging.info(f'Head Parameters: {args.head_type}')
    # logging.info(head_factory.head_param)
    model = FaceModel(num_classes=args.num_classes,multi_head=args.multi_head)
    # model = create_RepVGG_B0(num_classes=args.num_classes)
    # model = torchvision.models.resnet34(False,num_classes=args.num_classes)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        #logging.info(checkpoint.keys())
        # checkpoint_model = checkpoint
        checkpoint_model = checkpoint['model']
        checkpoint_model.pop('head.weight')
        checkpoint_model.pop('head.bias')
        model.load_state_dict(checkpoint_model,strict=False)
        logging.info(f"load pretrain :{args.finetune}")
    model.to(args.device)

    if args.sync_bn and args.DDP:
        #将模型中的批归一化层转换为同步批归一化层
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.gpu],find_unused_parameters=False)
    else:
        model = DP(model)
    #model 是使用 DataParallel 包装的模型，而 model.module 就是获取没有 DataParallel 包装的原始模型对象。
    #这样可以使得在代码中处理模型时更加方便，因为原始模型没有外层的包装器。在一些特定的操作中，可能需要直接访问原始模型的属性和方法，而不受 DataParallel 的影响。
    model_without_dp = model.module


    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logging.info(f'number of params:{n_parameters}')

    # better not to scale up lr for AdamW optimizer
    # linear_scaled_lr = args.lr + 5e-4 * args.batch_size * utils.get_world_size() / 1024.0
    # linear_scaled_lr = args.lr  * args.batch_size / 1024.0
    # args.lr = linear_scaled_lr
    #创建优化器并初始化
    optimizer = create_optimizer(args, model_without_dp)
    #创建损失缩放器对象‘loss_scaler’，在混合精度训练中用于缩放梯度值，以处理梯度溢出或者梯度下溢问题。
    #混合精度训练是一种优化训练过程的技术，其中输入数据和权重参数使用较低的数值精度（半精度浮点数），而梯度计算使用较高的精度（单精度浮点数）。
    #缩放器的作用是缩放梯度值，以适应不同精度的计算
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    # criterion = AMSoftmax(s=args.s, m=args.m)
    criterion = torch.nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)

    ema = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_dp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if args.ema and checkpoint.get('state_dict_ema', None) is not None:
                ema = ModelEma(model, device=args.device, resume=args.resume)
            else:
                ema = ModelEma(model, device=args.device)
    else:
        if args.ema:
            ema = ModelEma(model, device=args.device)
    if args.eval:
        test_stats = evaluate_test(data_loader_val, model, criterion, args.device)
        print(test_stats)
        logging.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc_bi']:.1f}%")
        return

    logging.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_TNR = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.DDP:
            data_loader_train.sampler.set_epoch(epoch)
        # 当epoch = args.warmup_epochs + 1 表示已经完成了预热阶段，进入了正式训练阶段。
        if epoch == args.warmup_epochs + 1:
            from copy import deepcopy
            #deepcopy 函数用于创建一个对象的深拷贝，即复制对象及其所有嵌套的对象，使得新的对象与原对象完全独立，修改其中一个对象不会影响另一个对象
            ema.ema = deepcopy(model)
            ema.ema.eval()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, args.device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, ema, None, weight=args.weight,bi_class_weight=args.bi_class_weight,s=1,RDrop=args.RDrop
        )
        # """BMMD loss"""
        # train_stats = train_one_epoch_inner(
        #     model, criterion, data_loader_train,
        #     optimizer, args.device, epoch, loss_scaler,
        #     args.clip_grad, args.clip_mode, ema, None, weight=args.weight,bi_class_weight=args.bi_class_weight,consistency_loss_weight=args.consistency_loss_weight,margin=args.bmmd_margin,RDrop=args.RDrop)

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.module.state_dict(),
                    'state_dict_ema': ema.ema.module.state_dict() if ema else None,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        if True:
            val_stats = evaluate(data_loader_val, model, criterion, args.device, bi_class_weight=args.bi_class_weight)
            test_stats = evaluate_test(data_loader_test, model, criterion, args.device, bi_class_weight=args.bi_class_weight)
            """BMMD"""
            # val_stats = evaluate_inner(data_loader_val, model, criterion, args.device, args.bi_class_weight,args.weight,args.consistency_loss_weight)
            # test_stats = evaluate_inner_test(data_loader_test, model, criterion, args.device)
            if ema and epoch > args.warmup_epochs:
                logging.info('  ************ EMA ************  ')
                val_ema_stats = evaluate_test(data_loader_val, ema.ema, criterion, args.device,bi_class_weight=args.bi_class_weight)
                test_ema_stats = evaluate_test(data_loader_test, ema.ema, criterion, args.device,bi_class_weight=args.bi_class_weight)
                # val_ema_stats = evaluate_inner(data_loader_val, ema.ema, criterion, args.device, args.bi_class_weight,args.weight,args.consistency_loss_weight)
                # test_ema_stats = evaluate_inner_test(data_loader_test, ema.ema, criterion, args.device)

            max_TNR = max(max_TNR, test_stats["tnr_0.002"]['TNR'],test_ema_stats["tnr_0.002"]['TNR'] if (ema and epoch > args.warmup_epochs) else 0)
            logging.info(f'Max TNR: {max_TNR:.5f}%')


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            if ema and epoch> args.warmup_epochs:
                log_stats.update({f'val_ema_{k}': v for k, v in val_ema_stats.items()})
                log_stats.update({f'test_ema_{k}': v for k, v in test_ema_stats.items()})

            ckpt=args.output_dir + f'/epoch_{epoch}.pth'
            #if max_TNR == max(test_stats["tnr_0.002"]['TNR'],test_ema_stats["tnr_0.002"]['TNR'] if ema and epoch > args.warmup_epochs else 0):
            if ema and epoch > args.warmup_epochs and test_stats["tnr_0.002"]['TNR'] < test_ema_stats["tnr_0.002"]['TNR']:
                utils.save_on_master({'model': ema.ema.module.state_dict()}, ckpt)
            else:
                utils.save_on_master({'model': model.module.state_dict()}, ckpt)

        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'EfficientFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = time.asctime().split()
    args.output_dir = f'run/{data[1]}_{data[2]}_RepVGG_MultiSoftmax'

    if not args.eval:
        args.output_dir += f'_198_weight{args.weight}_size{args.input_size}'
        if args.resume:
            args.output_dir = 'run/' + args.resume.strip().split('/')[1]
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(args.output_dir, 'log.txt')),
                logging.StreamHandler()
            ])
        shutil.copy('./main_HL.py', args.output_dir + '/main_HL.py')
        shutil.copy('./util/engine.py', args.output_dir + '/engine.py')
        shutil.copy('./util/losses.py', args.output_dir + '/losses.py')
        shutil.copy('./util/datasets.py', args.output_dir + '/datasets.py')
        print('*** Save to', args.output_dir)
        # if not args.resume:
        logging.info(args)
    main(args)
"""
#python main_HL.py  --gpus 0,1,2,3 --multi_head --ema --batch-size 1024 --consistency_loss_weight 1.0 --bmmd_margin 1.0 
python main_HL.py  --gpus 0,1,2,3,4,5,6,7 --multi_head --ema --batch-size 1024 --finetune /mnt/mfs2/haoyu.wang/project/repvgg_norm/run/Jul_6_RepVGG_MultiSoftmax_All_Dg_ClsW151_weight1.0_size224/epoch_91.pth
"""

