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
from timm.models.vision_transformer import Attention
from amsoftmax import AMSoftmax
import random
import numpy as np
import torch
import logging
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from head.AdM_Softmax import  ADM_Softmax
class FaceModel(torch.nn.Module):
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
        super(FaceModel, self).__init__()
        self.multi_head= multi_head
        self.backbone = create_RepVGG_B0(num_classes=0, use_checkpoint='saved_weight/RepVGG-B0-train.pth')
        self.max_pool = torch.nn.MaxPool2d(3,stride=2,padding=1)
        # self.conv1x1 = torch.nn.Conv1d(5,1,3,padding=1)
        # self.IN = nn.ModuleList([torch.nn.InstanceNorm1d(64),torch.nn.InstanceNorm1d(128),torch.nn.InstanceNorm1d(256),torch.nn.InstanceNorm1d(1280)])
        in_channels_list = [128,256,1280]
        out_dim = 512
        self.fpn = FPN(in_channels_list,out_dim)
        self.ssh = SSH(out_dim,out_dim)
        self.mil = MIL(out_dim)
        self.atten = Attention(out_dim,num_heads=8)
        # self.LN = nn.LayerNorm(512)
        self.LN = nn.InstanceNorm1d(512)
        torch.nn.init.trunc_normal_(self.mil.w,std=0.02)
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
            self.head_loc_binary = torch.nn.Linear(out_dim,2)
            torch.nn.init.xavier_uniform_(self.head_loc_binary.weight)
    def forward(self, data, mask):
        output = None
        loc_output = None
        if self.training:
            feat = self.backbone.forward_feature(data)
            feat_whole = feat[-1]
            feats = [feat[2],feat[3],feat[4]]
            feats = self.ssh(self.fpn(feats))
            masks = self.max_pool(self.max_pool(self.max_pool(mask)))
            feat_loc = torch.einsum('b m h w, b c h w -> b m c', masks, feats)# B,M,C
            if False in feat_loc.isfinite():
                print(f"feat_loc before is Nan , stopping training")
                sys.exit(1)
            feat_loc = self.atten(self.LN(feat_loc))
            # feat_loc = feat_loc/(masks.sum((-1,-2)).unsqueeze(-1)+1e-8)
            if False in feat_loc.isfinite():
                print(masks.sum((-1,-2)))
                print(f"feat_loc is Nan , stopping training")
                sys.exit(1)
            # for i in range(feat_loc.shape[1]):
            #     print(f'maks_{i},feature sum {feat_loc[0,i,:].sum()}')
            #     print(f'mask_{i}@feature:{torch.mm(feat_loc[0,i:i+1,:],self.mil.w).item()}')
            feat_loc, weights = self.mil(feat_loc) # B,C
            # print(self.mil.w.view(-1))
            # print(weights[0])
            # print(feat_loc[0].sum())
            # for i in range(feat_loc.shape[1]):
            #     print(f'maks {i},feature max {feat_loc[:,i,:].max(1)}')
            # feat_loc = feat_loc.mean(dim=1)
            # print(mask.shape),
            pred = self.head(feat_whole)
            if self.multi_head:
                output= self.head_binary(feat_whole)
                loc_output = self.head_loc_binary(feat_loc)
        else:
            feat_whole = self.backbone(data)
            pred = self.head(feat_whole)
            if self.multi_head:
                output = self.head_binary(feat_whole)
        return pred,output,loc_output
def get_args_parser():
    parser = argparse.ArgumentParser(
        'norm cls training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    # Dataset parameters
    parser.add_argument('--train_list', default='data_txts/8g4/huokai_train_PIL_resize224_positive.txt', type=str,help='dataset path')
    parser.add_argument('--train_add_list', default=['data_txts/8g4/huokai_train_PIL_resize224_crazytalknew_clean.txt'], type=list,help='add dataset path')
    # data_txts/8g8/huokai_train_PIL_resize224_crazytalknew_clean.txt'
    # /mnt/mfs2/ailun.li/data_txt/huokai/huokai_train.txts
    # data_txts/8g8/huokai_train_PIL_resize224_8g8.txt
    # data_txts/8g8/huokai_train_PIL_resize224_positive.txt,data_txts/8g8/huokai_train_PIL_resize224_crazytalk.txt, data_txts/8g8/huokai_train_PIL_resize224_muglife.txt
    parser.add_argument('--val_list', default='data_txts/8g4/huokai_test_PIL_resize224_positive.txt', type=str,help='dataset path')
    parser.add_argument('--val_add_list', default=['data_txts/8g4/huokai_test_PIL_resize224_crazytalknew_clean.txt'],type=list, help='add dataset path')
    # 'data_txts/8g8/huokai_test_PIL_resize224_crazytalknew_clean.txt'
    #/mnt/mfs2/ailun.li/data_txt/huokai/huokai_test.txts
    # data_txts/8g8/huokai_test_PIL_resize224_8g8.txt
    # data_txts/8g8/huokai_test_PIL_resize224_positive.txt,data_txts/8g8/huokai_test_PIL_resize224_crazytalk.txt, data_txts/8g8/huokai_test_PIL_resize224_muglife.txt
    # Model parameters
    parser.add_argument("--backbone_type", type = str,default='RepVGG', help = "Mobilefacenets, Resnet.")
    parser.add_argument("--backbone_conf_file", type = str, default='./backbone/backbone_conf.yaml', help = "the path of backbone_conf.yaml.")
    parser.add_argument("--multi_head", action='store_true', default=True,help="")
    parser.add_argument('--input-size', default=224,type=int, help='images input size')
    parser.add_argument('--num_classes', default=24,type=int, help='num classes')
    parser.add_argument('--s', default=1, type=int, help='num classes')
    parser.add_argument('--mw', default=1.0,type=float, help='multi classify weight')
    parser.add_argument('--bw', default=1., type=float, help='binary classify weight')
    parser.add_argument('--lw', default=1., type=float, help='multi classify weight')
    parser.add_argument('--bi_class_weight', default=[1,3], type=list, help="Binary classify weight")
    parser.add_argument('--RDrop', default=0.0, type=float, help="RDrop, Only dropout in Model")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--output_dir', default='./weight',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--gpus', default='0,1,2,3', type=str,help='num gpus')
    parser.add_argument('--ema', action='store_true', help='Model ema')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-6, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=10, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
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

from sbi.loc_dataset import Loc_Dataset
def main(args):
    if args.DDP:
        utils.init_distributed_mode(args)
    fix_random(args.seed)
    dataset_train = Loc_Dataset(args.train_list, phase='train', image_size=args.input_size)
    print('train',len(dataset_train.label_list))
    for tal in args.train_add_list:
        dataset_train_add = Loc_Dataset(tal, phase='train', image_size=args.input_size)
        print('add',len(dataset_train_add.label_list))
        dataset_train.image_list+= dataset_train_add.image_list
        dataset_train.label_list += dataset_train_add.label_list
    # dataset_val = Loc_Dataset(args.val_list, phase='test', image_size=args.input_size)
    # for val in args.val_add_list:
    #     dataset_val_add = Loc_Dataset(val, phase='test', image_size=args.input_size)
    #     dataset_val.image_list+=dataset_val_add.image_list
    #     dataset_val.label_list += dataset_val_add.label_list

    dataset_val = CustomDataset(args.val_list, type='test', transforms=get_transform('test',resize=args.input_size))
    for tal in args.val_add_list:
        dataset_val_add = CustomDataset(tal, type='test', transforms=None)
        dataset_val.imgs+=dataset_val_add.imgs
    logging.info(f'Train data:{len(dataset_train)}')
    logging.info(f'val data: {len(dataset_val)}')
    if args.DDP:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        logging.info(f"num tasks : {num_tasks}")
        logging.info(f"global_rank : {global_rank}")
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
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
    model = FaceModel(num_classes=args.num_classes,multi_head=args.multi_head)
    # model = create_RepVGG_B0(num_classes=args.num_classes)
    # model = torchvision.models.resnet34(False,num_classes=args.num_classes)
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model)
        logging.info(f"load pretrain :{args.finetune}")
    model.to(args.device)
    if args.sync_bn and args.DDP:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.gpu],find_unused_parameters=False)
    else:
        model = DP(model)
    model_without_dp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logging.info(f'number of params:{n_parameters}')

    # better not to scale up lr for AdamW optimizer
    # linear_scaled_lr = args.lr + 5e-4 * args.batch_size * utils.get_world_size() / 1024.0
    # linear_scaled_lr = args.lr  * args.batch_size / 1024.0
    # args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_dp)

    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    # criterion = AMSoftmax(s=args.s, m=args.m)
    criterion = torch.nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)
    ema = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_dp.load_state_dict(checkpoint['model'], strict=True)
        # model_without_ddp.load_state_dict(checkpoint, strict=False)
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
            ema = ModelEma(model,device=args.device)
    if args.eval:
        test_stats = evaluate_loc(data_loader_val, model, criterion, args.device, args.s, args.bi_class_weight)
        # score_f = sns.histplot(scores_dict, element = "poly")
        # score_f.get_figure().savefig(args.output_dir+'/score_distributions.png')
        # score_f.clear()
        print(test_stats)
        logging.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc_bi']:.1f}%")
        return
    logging.info(f"Start training for {args.start_epoch} epochs")
    start_time = time.time()
    max_TNR = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.DDP:
            data_loader_train.sampler.set_epoch(epoch)
        if epoch==args.warmup_epochs+1:
            from copy import deepcopy
            ema.ema = deepcopy(model)
            ema.ema.eval()
        train_stats = train_one_epoch_loc(
            model, criterion, data_loader_train,
            optimizer, args.device, epoch, loss_scaler,args.clip_grad, args.clip_mode, ema, None, weight=[args.mw,args.bw,args.lw],bi_class_weight=args.bi_class_weight,s=args.s,RDrop=args.RDrop)
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
            test_stats = evaluate_loc(data_loader_val, model, criterion, args.device, args.s, args.bi_class_weight)
            if ema and epoch> args.warmup_epochs:
                logging.info('  ************ EMA ************  ')
                test_ema_stats = evaluate_loc(data_loader_val, ema.ema, criterion, args.device, args.s, args.bi_class_weight)
            max_TNR = max(max_TNR, test_stats["tnr_0.002"]['TNR'],test_ema_stats["tnr_0.002"]['TNR'] if ema and epoch> args.warmup_epochs else 0)
            logging.info(f'Max accuracy: {max_TNR:.5f}%')
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            if ema and epoch> args.warmup_epochs:
                log_stats.update({f'test_ema_{k}': v for k, v in test_ema_stats.items()})
            ckpt=args.output_dir + f'/epoch_{epoch}.pth'

            if max_TNR == max(test_stats["tnr_0.002"]['TNR'],test_ema_stats["tnr_0.002"]['TNR'] if ema and epoch> args.warmup_epochs else 0):
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
    args.output_dir = f'run/{data[1]}_{data[2]}_{args.backbone_type}_LOC_FPNMIL_AttenIN_MultiSoftmax'
    if not args.eval:
        args.output_dir += f'_CrazytalkNewClean_NofilterLandmark_Dg_ClsW13_mw{args.mw}_bw{args.bw}_lw{args.lw}_size{args.input_size}_epoch{args.epochs}'
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
        shutil.copy('./main_loc_fpn.py', args.output_dir + '/main_loc_fpn.py')
        shutil.copy('./util/engine.py', args.output_dir + '/engine.py')
        shutil.copy('./sbi/loc_dataset.py', args.output_dir + '/loc_datasets.py')
        print('Save to', args.output_dir)
        if not args.resume:
            logging.info(args)
    else:
        args.output_dir = 'results/'+args.resume.strip().split('/')[1]
        print(args.output_dir)
    main(args)


"""
python main_loc_fpn.py --gpus 6,7 --batch-size 512 --ema
"""
