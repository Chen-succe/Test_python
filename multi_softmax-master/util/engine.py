# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import os.path

import math
import sys
from typing import Iterable, Optional

import torch
from sklearn.metrics import recall_score,precision_score,roc_auc_score
from timm.data import Mixup
from timm.utils import accuracy, ModelEma,AverageMeter

import util.utils as utils
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from util.trick import compute_kl_loss
from optimizer import *
from util.losses import *
import logging
from torchvision.utils import  save_image
from timm.utils.clip_grad import dispatch_clip_grad
from util import get_transform,CustomDataset
from torch.utils.data import DataLoader
import glob





def call_cls_loss(real_dist,fake_dist,bi_labels,bi_class_weight,scale):
    real_dist = torch.mean(real_dist, dim=1, keepdim=True)
    # with torch.no_grad():
    # fake_dist = torch.where(fake_dist>=real_dist,fake_dist,torch.zeros_like(real_dist))
    # fake_dist = torch.mean(fake_dist, dim=1, keepdim=True)
    # fake_dist*=mask
    fake_dist, _ = torch.max(fake_dist, dim=1, keepdim=True)
    # fake_dist = fake_dist.sum(1,keepdim=True)/(mask.sum(1,keepdim=True)+1e-8)
    dist = torch.cat((real_dist, fake_dist), dim=1)
    # cls_loss = torch.nn.functional.cross_entropy(dist,bi_labels)
    probs = F.softmax(dist.detach().clone()*scale, dim=1)
    # probs_log = torch.log(probs)
    # cls_loss = F.nll_loss(probs_log, bi_labels)
    cls_loss = F.cross_entropy(dist*scale, bi_labels, bi_class_weight)
    return cls_loss,probs

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, weight=1.0,bi_class_weight=[1,1],s=1, RDrop = 0.):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    TP,FP,TN,FN = 0,0,0,0

    for samples, targets in metric_logger.log_every(
            tqdm(data_loader), print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = len(targets)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # print(f'Len of data:{len(samples)}')
        with torch.cuda.amp.autocast():
            outputs,binary_outputs = model(samples)
            m_loss = criterion(outputs, targets)
            bi_labels = targets.clone()
            bi_labels[bi_labels < 1] = 0
            bi_labels[bi_labels >= 1] = 1
            bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
            if model.module.multi_head:
                # binary_outputs *=s # need to scale up output if binary head norm feature
                probs = F.softmax(binary_outputs.clone().detach(), dim=1)
                cls_loss = F.cross_entropy(binary_outputs, bi_labels, bi_class_weight)
            else:
                real_dist = outputs[:, :1]
                fake_dist = outputs[:, 1:]
                cls_loss,probs = call_cls_loss(real_dist,fake_dist,bi_labels,bi_class_weight,s)
            loss = cls_loss + weight * m_loss
            if RDrop:
                outputs_, binary_outputs_ = model(samples)
                kl_loss_mul = compute_kl_loss(outputs, outputs_)
                loss += RDrop * kl_loss_mul
                kl_loss_bi = compute_kl_loss(binary_outputs, binary_outputs_)
                # print(kl_loss_mul,kl_loss_bi)
                loss += RDrop * kl_loss_bi
        # metric
        loss_value = loss.item()
        acc_multi = accuracy(outputs, targets)
        acc_bi = accuracy(probs,bi_labels)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        P = TP / (TP + FP + 1e-8)
        R = TP / (TP + FN + 1e-8)
        F1 = 2 * P * R / (P + R + 1e-8)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print('binary_outputs')
            print(binary_outputs.max())
            print(binary_outputs.min())
            sys.exit(1)
        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)
        # torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        metric_logger.meters['P'].update(P, n=bs)
        metric_logger.meters['R'].update(R, n=bs)
        metric_logger.meters['F1'].update(F1, n=bs)
        metric_logger.meters['loss_multi'].update(m_loss.item(),n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
        if RDrop:
            metric_logger.meters['loss_RDrop'].update((kl_loss_mul+kl_loss_bi).item(), n=bs)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, criterion, device, scale=1,bi_class_weight=[1,1], weight=1.0):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    # switch to evaluation mode
    model.eval()
    pos_list, neg_list,label_list= [], [],[]
    TP,FP,TN,FN = 0,0,0,0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs, binary_outputs = model(images)
        m_loss = criterion(outputs, targets)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
        if model.module.multi_head:
            # binary_outputs *= scale # need to scale up output if binary head norm feature
            probs = F.softmax(binary_outputs, dim=1)
            cls_loss = F.cross_entropy(binary_outputs,bi_labels,bi_class_weight)
        #      probs_log = torch.log(probs)
        #      bi_class_weight = torch.tensor(bi_class_weight).to(probs_log)
        #      cls_loss = F.nll_loss(probs_log, bi_labels, weight=bi_class_weight)
        else:
            real_dist = outputs[:, :1]
            fake_dist = outputs[:, 1:]
            cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, bi_class_weight, model.module.head.scale)

        # cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, 1)
        # real_dist = torch.mean(real_dist, dim=1, keepdim=True)
        # fake_dist, _ = torch.max(fake_dist, dim=1, keepdim=True)
        # # fake_dist = torch.mean(fake_dist, dim=1, keepdim=True)
        # dist = torch.cat((real_dist, fake_dist), dim=1)
        # probs = F.softmax(scale * dist, dim=1)
        # probs = F.softmax(dist, dim=1)
        # probs_log = torch.log(probs)
        # cls_loss = F.nll_loss(probs_log, bi_labels)

        loss = cls_loss + weight * m_loss
        scores = probs[:, 1].tolist()
        for i in range(len(targets)):
            if targets[i] == 0:
                pos_list.append(scores[i])
            else:
                neg_list.append(scores[i])
        acc_bi = accuracy(probs, bi_labels)
        acc_multi = accuracy(outputs,targets)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        bs = images.shape[0]
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        metric_logger.meters['loss_multi'].update(m_loss.item(), n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)
    test_stat['P'] = P
    test_stat['R'] = R
    test_stat["F1"] = F1
    # bi_labels = [0]*len(pos_list)+[1]*len(neg_list)
    # auc = roc_auc_score(bi_labels,pos_list+neg_list)
    # test_stat["AUC"] = auc
    # print('* Acc@multi {acc_multi.global_avg:.3f} Acc@bi {acc_bi.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(acc_multi=metric_logger.acc_multi, acc_bi=metric_logger.acc_bi, losses=metric_logger.loss))
    # print("out stat: ", dist.mean().item(), dist.std().item())
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)).sort()[0]
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)).sort()[0]
    print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')

    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        # print(all_pos_list[int(len(all_pos_list) * (1-threshold_r)) + 1:])
        test_stat[f'tnr_{threshold_r}'] = {
            'TNR':neg_cnt.item() / len(all_neg_list),
            'th':thresh.item()
        }
    print("Averaged stats:", test_stat)
    return test_stat




@torch.no_grad()
def evaluate_test(data_loader, model, criterion, device, scale=0,bi_class_weight=[1,1], weight=1.0):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    size = 224
    transform_test = get_transform('test',size)
    pos_list, neg_list = [], []
    TP,FP,TN,FN = 0,0,0,0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs, binary_outputs = model(images)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        if model.module.multi_head:
            probs = F.softmax(binary_outputs, dim=1)
        else:
            real_dist = outputs[:, :1]
            fake_dist = outputs[:, 1:]
            cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, bi_class_weight, model.module.head.scale)
        scores = probs[:, 1].tolist()
        for i in range(len(targets)):
            if targets[i] == 0:
                pos_list.append(scores[i])
            else:
                neg_list.append(scores[i])
        acc_bi = accuracy(probs, bi_labels)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        bs = images.shape[0]
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # P = TP / (TP + FP + 1e-8)
    # R = TP / (TP + FN + 1e-8)
    # F1 = 2 * P * R / (P + R + 1e-8)
    # test_stat['precision'] = P
    # test_stat['recall'] = R
    # test_stat["F1"] = F1
    # bi_labels = [0]*len(pos_list)+[1]*len(neg_list)
    # auc = roc_auc_score(bi_labels,pos_list+neg_list)
    # test_stat["AUC"] = auc
    # print("out stat: ", dist.mean().item(), dist.std().item())
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)).sort()[0]
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)).sort()[0]
    print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')
    threshold_rs = [0.0001, 0.0002]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        tnr_info = {
            'TNR': neg_cnt.item() / len(all_neg_list),
            'th': thresh.item(),
        }
        print(f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        test_stat[f'tnr_{threshold_r}'] = tnr_info
        #test_stat[f'FNR_{threshold_r}'] = {f'FNR_{path}': value for path, value in test_stat.items() if path.startswith('FNR_')}

    return test_stat


def train_one_epoch_sam(model: torch.nn.Module, criterion,
                    data_loader: Iterable, sam_optimizer: SAM,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, weight=1.0,bi_class_weight=[1,1],s=30, RDrop = 0.):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    TP,FP,TN,FN = 0,0,0,0

    for samples, targets in metric_logger.log_every(
            tqdm(data_loader), print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        #
        if not math.isfinite(torch.sum(samples)):
            print("Image is {}, stopping training".format(torch.sum(samples)))
            continue
        bs = len(targets)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        enable_running_stats(model)
        outputs,binary_outputs = model(samples,targets)
        m_loss = criterion(outputs, targets)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
        if model.module.multi_head:
            probs = F.softmax(binary_outputs.detach().clone(), dim=1)
            cls_loss = F.cross_entropy(binary_outputs, bi_labels, bi_class_weight)
        else:
            real_dist = outputs[:, :1]
            fake_dist = outputs[:, 1:]
            cls_loss,probs = call_cls_loss(real_dist,fake_dist,bi_labels,bi_class_weight,model.module.head.scale)

        loss = cls_loss + weight * m_loss
        if RDrop:
            outputs_, binary_outputs_ = model(samples, targets)
            kl_loss_mul = compute_kl_loss(outputs, outputs_)
            loss += RDrop * kl_loss_mul
            kl_loss_bi = compute_kl_loss(binary_outputs, binary_outputs_)
            # print(kl_loss_mul,kl_loss_bi)
            loss += RDrop * kl_loss_bi
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print('binary_outputs')
            print(binary_outputs.max())
            print(binary_outputs.min())
            sys.exit(1)
        loss.backward()
        sam_optimizer.first_step(zero_grad=True)
        disable_running_stats(model)
        outputs, binary_outputs = model(samples, targets)
        m_loss = criterion(outputs, targets)
        if model.module.multi_head:
            probs = F.softmax(binary_outputs.detach().clone(), dim=1)
            # probs_log = torch.log(probs)
            cls_loss = F.cross_entropy(binary_outputs, bi_labels, bi_class_weight)
            # cls_loss = F.nll_loss(probs_log, bi_labels,weight=bi_class_weight)
        else:
            real_dist = outputs[:, :1]
            fake_dist = outputs[:, 1:]
            cls_loss,probs = call_cls_loss(real_dist,fake_dist,bi_labels,bi_class_weight,model.module.head.scale)
        loss = cls_loss + weight * m_loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print('binary_outputs')
            print(binary_outputs.max())
            print(binary_outputs.min())
            sys.exit(1)
        loss.backward()
        sam_optimizer.second_step(zero_grad=True)

        # metric
        acc_multi = accuracy(outputs, targets)
        acc_bi = accuracy(probs,bi_labels)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        P = TP / (TP + FP + 1e-8)
        R = TP / (TP + FN + 1e-8)
        F1 = 2 * P * R / (P + R + 1e-8)

        # this attribute is added by timm on one optimizer (adahessian)
        # is_second_order = hasattr(
        #     optimizer, 'is_second_order') and optimizer.is_second_order
        # loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
        #             parameters=model.parameters(), create_graph=is_second_order)
        # torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        metric_logger.meters['precision'].update(P, n=bs)
        metric_logger.meters['recall'].update(R, n=bs)
        metric_logger.meters['f1'].update(F1, n=bs)
        metric_logger.meters['loss'].update(loss_value,n=bs)
        # metric_logger.update(loss=loss_value,)
        metric_logger.update(lr=sam_optimizer.base_optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


"""
train or evaluate with SBI loss
"""
def train_one_epoch_sbi(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, weight=1.0,bi_class_weight=[1,1],s=30, RDrop = 0.):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    TP,FP,TN,FN = 0,0,0,0

    for samples, targets in metric_logger.log_every(
            tqdm(data_loader), print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = len(targets)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            targets[targets>0] = 1

            bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
            # bi_class_weight[0] = (targets==1).sum()/(targets==0).sum()
            loss = F.cross_entropy(outputs,targets,weight=bi_class_weight)
            probs  = torch.softmax(outputs.clone().detach(),dim=1)


            """  """
            # outputs,binary_outputs = model(samples,targets)
            # m_loss = criterion(outputs, targets)
            # bi_labels = targets.clone()
            # bi_labels[bi_labels < 1] = 0
            # bi_labels[bi_labels >= 1] = 1
            # bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
            # if model.module.multi_head:
            #     probs = F.softmax(binary_outputs.clone().detach(), dim=1)
            #     # probs_log = torch.log(probs)
            #     cls_loss = F.cross_entropy(binary_outputs, bi_labels, bi_class_weight)
            #     # cls_loss = F.nll_loss(probs_log, bi_labels,weight=bi_class_weight)
            # else:
            #     real_dist = outputs[:, :1]
            #     fake_dist = outputs[:, 1:]
            #     cls_loss,probs = call_cls_loss(real_dist,fake_dist,bi_labels,bi_class_weight,model.module.head.scale)

            # print(probs_log.max(),probs_log.min())
            # print(cls_loss,m_loss)
            # loss = cls_loss + weight * m_loss
            # if RDrop:
            #     outputs_, binary_outputs_ = model(samples, targets)
            #     kl_loss_mul = compute_kl_loss(outputs, outputs_)
            #     loss += RDrop * kl_loss_mul
            #     kl_loss_bi = compute_kl_loss(binary_outputs, binary_outputs_)
            #     # print(kl_loss_mul,kl_loss_bi)
            #     loss += RDrop * kl_loss_bi
        # metric
        bi_labels = targets.clone()
        loss_value = loss.item()
        acc_multi = accuracy(outputs, targets)
        acc_bi = accuracy(probs,bi_labels)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        P = TP / (TP + FP + 1e-8)
        R = TP / (TP + FN + 1e-8)
        F1 = 2 * P * R / (P + R + 1e-8)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        # torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        metric_logger.meters['precision'].update(P, n=bs)
        metric_logger.meters['recall'].update(R, n=bs)
        metric_logger.meters['f1'].update(F1, n=bs)
        metric_logger.meters['loss'].update(loss_value,n=bs)
        # metric_logger.update(loss=loss_value,)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluate_sbi(data_loader, model, criterion, device, scale=1,bi_class_weight=[1,1], weight=1.0):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    # switch to evaluation mode
    model.eval()
    pos_list, neg_list,label_list= [], [],[]
    TP,FP,TN,FN = 0,0,0,0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
        targets[targets>0] = 1
        loss = F.cross_entropy(outputs,targets,bi_class_weight)
        probs = torch.softmax(outputs, dim=1)
        bi_labels = targets.clone()

        # outputs, binary_outputs = model(images, targets)
        # loss = criterion(outputs, targets)
        # bi_labels = targets.clone()
        # bi_labels[bi_labels < 1] = 0
        # bi_labels[bi_labels >= 1] = 1
        # bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
        # if model.module.multi_head:
        #     # binary_outputs *= scale # need to scale up output if binary head norm feature
        #     probs = F.softmax(binary_outputs, dim=1)
        #     cls_loss = F.cross_entropy(binary_outputs,bi_labels,bi_class_weight)
        # #      probs_log = torch.log(probs)
        # #      bi_class_weight = torch.tensor(bi_class_weight).to(probs_log)
        # #      cls_loss = F.nll_loss(probs_log, bi_labels, weight=bi_class_weight)
        # else:
        #     real_dist = outputs[:, :1]
        #     fake_dist = outputs[:, 1:]
        #     cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, bi_class_weight, model.module.head.scale)

        # cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, 1)
        # real_dist = torch.mean(real_dist, dim=1, keepdim=True)
        # fake_dist, _ = torch.max(fake_dist, dim=1, keepdim=True)
        # # fake_dist = torch.mean(fake_dist, dim=1, keepdim=True)
        # dist = torch.cat((real_dist, fake_dist), dim=1)
        # probs = F.softmax(scale * dist, dim=1)
        # probs = F.softmax(dist, dim=1)
        # probs_log = torch.log(probs)
        # cls_loss = F.nll_loss(probs_log, bi_labels)

        # loss = cls_loss + weight * loss

        scores = probs[:, 1].tolist()
        for i in range(len(targets)):
            if targets[i] == 0:
                pos_list.append(scores[i])
            else:
                neg_list.append(scores[i])
        acc_bi = accuracy(probs, bi_labels)
        acc_multi = accuracy(outputs,targets)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        bs = images.shape[0]
        metric_logger.meters['loss'].update(loss.item(),n=bs)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)
    test_stat['precision'] = P
    test_stat['recall'] = R
    test_stat["F1"] = F1
    bi_labels = [0]*len(pos_list)+[1]*len(neg_list)
    auc = roc_auc_score(bi_labels,pos_list+neg_list)
    test_stat["AUC"] = auc
    print('* Acc@multi {acc_multi.global_avg:.3f} Acc@bi {acc_bi.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(acc_multi=metric_logger.acc_multi, acc_bi=metric_logger.acc_bi, losses=metric_logger.loss))
    # print("out stat: ", dist.mean().item(), dist.std().item())
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)).sort()[0]
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)).sort()[0]
    print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')
    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(
            f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        # print(all_pos_list[int(len(all_pos_list) * (1-threshold_r)) + 1:])
        test_stat[f'tnr_{threshold_r}'] = {
            'TNR':neg_cnt.item() / len(all_neg_list),
            'th':thresh.item()
        }

    return test_stat




"""
train or evaluate with One-class loss
"""
# def train_one_epoch_oc(model: torch.nn.Module, criterion,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,
#                     clip_grad: float = 0,
#                     clip_mode: str = 'norm',
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, weight=1.0,bi_class_weight=[1,1],s=1, RDrop = 0.):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(
#         window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 100
#     TP,FP,TN,FN = 0,0,0,0
#     for samples, targets in metric_logger.log_every(
#             tqdm(data_loader), print_freq, header):
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)
#         if not math.isfinite(torch.sum(samples)):
#             print("Image is {}, stopping training".format(torch.sum(samples)))
#             continue
#         bs = len(targets)
#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
#         # print(f'Len of data:{len(samples)}')
#         with torch.cuda.amp.autocast():
#             outputs,binary_outputs = model(samples,targets.clone())
#             m_loss = criterion(outputs, targets)
#             bi_labels = targets.clone()
#             bi_labels[bi_labels >= 1] = 1
#             bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
#             if model.module.multi_head:
#                 # binary_outputs *=s # need to scale up output if binary head norm feature
#                 probs = (1- binary_outputs.clone().detach())/2
#                 cls_loss = F.softplus(binary_outputs).mean()
#             else:
#                 real_dist = outputs[:, :1]
#                 fake_dist = outputs[:, 1:]
#                 cls_loss,probs = call_cls_loss(real_dist,fake_dist,bi_labels,bi_class_weight,s)
#
#             # print(probs_log.max(),probs_log.min())
#             # print(cls_loss,m_loss)
#             loss = cls_loss + weight * m_loss
#             if RDrop:
#                 outputs_, binary_outputs_ = model(samples, targets)
#                 kl_loss_mul = compute_kl_loss(outputs, outputs_)
#                 loss += RDrop * kl_loss_mul
#                 kl_loss_bi = compute_kl_loss(binary_outputs, binary_outputs_)
#                 # print(kl_loss_mul,kl_loss_bi)
#                 loss += RDrop * kl_loss_bi
#         # metric
#         loss_value = loss.item()
#         acc_multi = accuracy(outputs, targets)
#         pred_labels_bi = probs.clone()
#         TP += ((pred_labels_bi <= 0.5) & (bi_labels == 0)).sum().item()
#         FP += ((pred_labels_bi <= 0.5 ) & (bi_labels == 1)).sum().item()
#         FN += ((pred_labels_bi > 0.5) & (bi_labels == 0)).sum().item()
#         TN += ((pred_labels_bi > 0.5) & (bi_labels == 1)).sum().item()
#         P = TP / (TP + FP + 1e-8)
#         R = TP / (TP + FN + 1e-8)
#         F1 = 2 * P * R / (P + R + 1e-8)
#         acc_bi = (TP+TN) / (TP + FN + TN + FP + 1e-8) * 100.
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print('binary_outputs')
#             print(binary_outputs.max())
#             print(binary_outputs.min())
#             sys.exit(1)
#
#         optimizer.zero_grad()
#
#         # this attribute is added by timm on one optimizer (adahessian)
#         is_second_order = hasattr(
#             optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
#                     parameters=model.parameters(), create_graph=is_second_order)
#
#         # torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)
#         metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
#         metric_logger.meters['acc_bi'].update(acc_bi, n=bs)
#         metric_logger.meters['precision'].update(P, n=bs)
#         metric_logger.meters['recall'].update(R, n=bs)
#         metric_logger.meters['f1'].update(F1, n=bs)
#         metric_logger.meters['loss'].update(loss_value,n=bs)
#         # metric_logger.update(loss=loss_value,)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# @torch.no_grad()
# def evaluate_oc(data_loader, model, criterion, device, scale=1,bi_class_weight=[1,1], weight=1.0):
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Val:'
#     # switch to evaluation mode
#     model.eval()
#     pos_list, neg_list,label_list= [], [],[]
#     TP,FP,TN,FN = 0,0,0,0
#     for images, targets in metric_logger.log_every(data_loader, 10, header):
#         images = images.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)
#
#         outputs, binary_outputs = model(images, targets.clone())
#         loss = criterion(outputs, targets)
#         bi_labels = targets.clone()
#         bi_labels[bi_labels < 1] = 0
#         bi_labels[bi_labels >= 1] = 1
#         bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
#         if model.module.multi_head:
#             probs = (1 - binary_outputs.clone().detach()) / 2
#             cls_loss = F.softplus(binary_outputs).mean()
#         else:
#             real_dist = outputs[:, :1]
#             fake_dist = outputs[:, 1:]
#             cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, bi_class_weight, model.module.head.scale)
#
#         loss = cls_loss + weight * loss
#         scores = probs[:,0].tolist()
#         for i in range(len(targets)):
#             if targets[i] == 0:
#                 pos_list.append(scores[i])
#             else:
#                 neg_list.append(scores[i])
#         acc_multi = accuracy(outputs,targets)
#         pred_labels_bi = probs.clone()
#         TP += ((pred_labels_bi <= 0.5) & (bi_labels == 0)).sum().item()
#         FP += ((pred_labels_bi <= 0.5 ) & (bi_labels == 1)).sum().item()
#         FN += ((pred_labels_bi > 0.5) & (bi_labels == 0)).sum().item()
#         TN += ((pred_labels_bi > 0.5) & (bi_labels == 1)).sum().item()
#         bs = images.shape[0]
#         metric_logger.meters['loss'].update(loss.item(),n=bs)
#         metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
#         # metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
#         # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#     metric_logger.synchronize_between_processes()
#     test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     P = TP / (TP + FP + 1e-8)
#     R = TP / (TP + FN + 1e-8)
#     F1 = 2 * P * R / (P + R + 1e-8)
#     acc_bi = (TP + TN) / (TP + FN + TN + FP + 1e-8) * 100.
#     test_stat["acc_bi"] = acc_bi
#     test_stat['precision'] = P
#     test_stat['recall'] = R
#     test_stat["F1"] = F1
#     bi_labels = [0]*len(pos_list)+[1]*len(neg_list)
#     auc = roc_auc_score(bi_labels,pos_list+neg_list)
#     test_stat["AUC"] = auc
#     print('* Acc@multi {acc_multi.global_avg:.3f} Acc@bi {acc_bi:.3f} loss {losses.global_avg:.3f}'
#           .format(acc_multi=metric_logger.acc_multi, acc_bi=acc_bi, losses=metric_logger.loss))
#     # print("out stat: ", dist.mean().item(), dist.std().item())
#     all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)).sort()[0]
#     all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)).sort()[0]
#     print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')
#
#     threshold_rs = [0.002, 0.005]
#     for threshold_r in threshold_rs:
#         threshold = int(len(all_pos_list) * threshold_r) + 1
#         thresh = all_pos_list.topk(threshold)[0][-1]
#         pos_cnt = torch.sum(all_pos_list > thresh)
#         neg_cnt = torch.sum(all_neg_list > thresh)
#         print(
#             f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
#         # print(all_pos_list[int(len(all_pos_list) * (1-threshold_r)) + 1:])
#         test_stat[f'tnr_{threshold_r}'] = {
#             'TNR':neg_cnt.item() / len(all_neg_list),
#             'th':thresh.item()
#         }
#
#     return test_stat
#
# #
"""
train or evaluate with inner loss
"""
def train_one_epoch_inner(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, weight=1.0,bi_class_weight=[1,1],consistency_loss_weight=1.0,margin=0, RDrop = 0.):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    TP,FP,TN,FN = 0,0,0,0
    # feature_consistency_loss = Inner_loss()
    # MMD = MMD_loss()
    BMMD = BatchMMD_loss(margin=margin)
    # Centerloss = CenterLoss(24,feat_dim=1280)
    # optimizer_center = torch.optim.SGD(Centerloss.parameters(), lr=0.5)
    # scaler = torch.cuda.amp.GradScaler()
    for samples, targets in metric_logger.log_every(
            tqdm(data_loader), print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = len(targets)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            feature,outputs,binary_outputs = model(samples)
            m_loss = criterion(outputs, targets)
            bi_labels = targets.clone()
            bi_labels[bi_labels < 1] = 0
            bi_labels[bi_labels >= 1] = 1
            # inner_loss = feature_consistency_loss(feat, bi_labels)
            bmmd_loss = BMMD(feature, targets.clone())
            # bmmd_loss = BMMD(feature,targets.clone(),bi_labels.clone())
            # center_loss  = Centerloss(feature,targets.clone())

            bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
            if model.module.multi_head:
                # binary_outputs *=s # need to scale up output if binary head norm feature
                probs = F.softmax(binary_outputs.clone().detach(), dim=1)
                cls_loss = F.cross_entropy(binary_outputs, bi_labels, bi_class_weight)
            else:
                real_dist = outputs[:, :1]
                fake_dist = outputs[:, 1:]
                cls_loss,probs = call_cls_loss(real_dist,fake_dist,bi_labels,bi_class_weight,1)

            loss = cls_loss + weight * m_loss + consistency_loss_weight * bmmd_loss # bmmd_loss,center_loss
            if RDrop:
                feature_,outputs_, binary_outputs_ = model(samples)
                # mmd_loss = MMD(feature,feature_)
                # loss += RDrop*mmd_loss
                # kl_loss_mul = compute_kl_loss(outputs, outputs_)
                # loss += RDrop * kl_loss_mul
                # kl_loss_bi = compute_kl_loss(binary_outputs, binary_outputs_)
                # loss += RDrop * kl_loss_bi

        loss_value = loss.item()
        acc_multi = accuracy(outputs, targets)
        acc_bi = accuracy(probs,bi_labels)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        P = TP / (TP + FP + 1e-8)
        R = TP / (TP + FN + 1e-8)
        F1 = 2 * P * R / (P + R + 1e-8)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print('binary_outputs')
            print(binary_outputs.max())
            print(binary_outputs.min())
            sys.exit(1)

        optimizer.zero_grad()
        # optimizer_center.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)
        """center loss backward and steop"""
        # scaler.scale(loss).backward(create_graph=False)
        # if clip_grad is not None:
        #     scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            # dispatch_clip_grad(model.parameters(), clip_grad, mode=clip_mode)
        # scaler.step(optimizer)
        # for param in Centerloss.parameters():
        #     param.grad.data *= (1. / consistency_loss_weight)
        # scaler.step(optimizer_center)
        # scaler.update()
        """"""
        # torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        metric_logger.meters['loss_multi'].update(m_loss.item(),n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
        # metric_logger.meters['loss_mmd'].update(mmd_loss.item(), n=bs)
        metric_logger.meters['loss_bmmd'].update(bmmd_loss.item(), n=bs)
        # metric_logger.meters['loss_center'].update(center_loss.item(), n=bs)
        metric_logger.meters['P'].update(P, n=bs)
        metric_logger.meters['R'].update(R, n=bs)
        metric_logger.meters['f1'].update(F1, n=bs)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluate_inner(data_loader, model, criterion, device, bi_class_weight=[1,1], weight=1.0,consistency_loss_weight=1.0,margin=0):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    # switch to evaluation mode
    model.eval()
    pos_list, neg_list,label_list= [], [],[]
    # feature_consistency_loss = Inner_loss()
    BMMD = BatchMMD_loss(margin=margin)
    # Centerloss = CenterLoss(24, feat_dim=1280)
    TP,FP,TN,FN = 0,0,0,0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        feature,outputs,binary_outputs = model(images)
        m_loss = criterion(outputs, targets)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        # inner_loss = feature_consistency_loss(feat, bi_labels)
        bmmd_loss = BMMD(feature,targets.clone())
        bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
        if model.module.multi_head:
            probs = F.softmax(binary_outputs, dim=1)
            cls_loss = F.cross_entropy(binary_outputs,bi_labels,bi_class_weight)
        else:
            real_dist = outputs[:, :1]
            fake_dist = outputs[:, 1:]
            cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, bi_class_weight, model.module.head.scale)
        # loss = cls_loss + weight * m_loss+inner_loss
        scores = probs[:, 1].tolist()
        for i in range(len(targets)):
            if targets[i] == 0:
                pos_list.append(scores[i])
            else:
                neg_list.append(scores[i])
        acc_bi = accuracy(probs, bi_labels)
        acc_multi = accuracy(outputs,targets)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        bs = images.shape[0]
        metric_logger.meters['loss_multi'].update(m_loss.item(),n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
        metric_logger.meters['loss_bmmd'].update(bmmd_loss.item(), n=bs)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
    metric_logger.synchronize_between_processes()
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)
    test_stat['P'] = P
    test_stat['R'] = R
    test_stat["F1"] = F1
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)).sort()[0]
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)).sort()[0]
    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(
            f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        test_stat[f'tnr_{threshold_r}'] = {
            'TNR':neg_cnt.item() / len(all_neg_list),
            'th':thresh.item()
        }
    return test_stat





@torch.no_grad()
def evaluate_inner_test(data_loader, model, criterion, device, bi_class_weight=[1,1]):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    pos_list, neg_list = [], []
    TP,FP,TN,FN = 0,0,0,0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        feature,outputs,binary_outputs = model(images)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        if model.module.multi_head:
            probs = F.softmax(binary_outputs, dim=1)
        else:
            real_dist = outputs[:, :1]
            fake_dist = outputs[:, 1:]
            cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, bi_class_weight, model.module.head.scale)
        scores = probs[:, 1].tolist()
        for i in range(len(targets)):
            if targets[i] == 0:
                pos_list.append(scores[i])
            else:
                neg_list.append(scores[i])
        acc_bi = accuracy(probs, bi_labels)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        bs = images.shape[0]
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
    metric_logger.synchronize_between_processes()
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)).sort()[0]
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)).sort()[0]
    print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')
    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        test_stat[f'tnr_{threshold_r}'] = {
            'TNR':neg_cnt.item() / len(all_neg_list),
            'th':thresh.item()
        }
    return test_stat




from xbm.xbm import XBM
def train_one_epoch_xbm(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, weight=1.0,bi_class_weight=[1,1],consistency_loss_weight=1.0,margin=0.5,xbm_helper=None, xbm_enable=False,xbm_start=2000,xbm_weight = 1):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    TP,FP,TN,FN = 0,0,0,0
    metric_LOSS = ContrastiveLoss(margin)
    total_step = len(data_loader)
    step = 0
    for samples, targets in metric_logger.log_every(
            tqdm(data_loader), print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = len(targets)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            feature,outputs,binary_outputs = model(samples)
            m_loss = criterion(outputs, targets)
            bi_labels = targets.clone()
            bi_labels[bi_labels < 1] = 0
            bi_labels[bi_labels >= 1] = 1

            batch_loss,xbm_loss = 0,0
            cur_step = epoch * total_step+step
            # if not (xbm_enable and cur_step > xbm_start):
            if consistency_loss_weight >0:
                batch_loss = metric_LOSS(feature,targets,feature,targets)
                if xbm_enable and cur_step > xbm_start:
                    xbm_helper.enqueue_dequeue(feature.detach().clone(), targets.detach().clone())
                    xbm_feats, xbm_targets = xbm_helper.get()
                    xbm_loss = metric_LOSS(feature, targets, xbm_feats, xbm_targets)

            bi_class_weight = torch.tensor(bi_class_weight).to(outputs)

            probs = F.softmax(binary_outputs.clone().detach(), dim=1)
            cls_loss = F.cross_entropy(binary_outputs, bi_labels, bi_class_weight)
            loss = cls_loss + weight * m_loss + consistency_loss_weight *(batch_loss + xbm_weight * xbm_loss) # bmmd_loss,center_loss

        loss_value = loss.item()
        acc_multi = accuracy(outputs, targets)
        acc_bi = accuracy(probs,bi_labels)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        P = TP / (TP + FP + 1e-8)
        R = TP / (TP + FN + 1e-8)
        F1 = 2 * P * R / (P + R + 1e-8)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print('m_loss',m_loss)
            print('binary loss',cls_loss)
            print('batch_loss',batch_loss)
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)
        # torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        metric_logger.meters['loss_multi'].update(m_loss.item(),n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
        if consistency_loss_weight >0:
            metric_logger.meters['loss_batch'].update(batch_loss.item(), n=bs)
        if xbm_enable and cur_step > xbm_start:
            metric_logger.meters['loss_xbm'].update(xbm_loss.item(), n=bs)
        metric_logger.meters['P'].update(P, n=bs)
        metric_logger.meters['R'].update(R, n=bs)
        metric_logger.meters['f1'].update(F1, n=bs)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        step += 1
        # torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluate_xbm(data_loader, model, criterion, device, bi_class_weight=[1,1],margin=0.5,xbm_helper=None, xbm_enable=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    # switch to evaluation mode
    model.eval()
    pos_list, neg_list,label_list= [], [],[]
    metric_LOSS = ContrastiveLoss(margin)
    TP,FP,TN,FN = 0,0,0,0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        feature,outputs,binary_outputs = model(images)
        m_loss = criterion(outputs, targets)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        batch_loss = metric_LOSS(feature, targets, feature, targets)
        if xbm_enable:
            xbm_feats, xbm_targets = xbm_helper.get()
            xbm_loss = metric_LOSS(feature, targets, xbm_feats, xbm_targets)
        bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
        probs = F.softmax(binary_outputs, dim=1)
        cls_loss = F.cross_entropy(binary_outputs,bi_labels,bi_class_weight)
        scores = probs[:, 1].tolist()
        for i in range(len(targets)):
            if targets[i] == 0:
                pos_list.append(scores[i])
            else:
                neg_list.append(scores[i])
        acc_bi = accuracy(probs, bi_labels)
        acc_multi = accuracy(outputs,targets)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        bs = images.shape[0]
        metric_logger.meters['loss_multi'].update(m_loss.item(),n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
        metric_logger.meters['loss_batch'].update(batch_loss.item(), n=bs)
        if xbm_enable:
            metric_logger.meters['loss_xbm'].update(xbm_loss.item(), n=bs)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
    metric_logger.synchronize_between_processes()
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)
    test_stat['P'] = P
    test_stat['R'] = R
    test_stat["F1"] = F1
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)).sort()[0]
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)).sort()[0]
    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(
            f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        test_stat[f'tnr_{threshold_r}'] = {
            'TNR':neg_cnt.item() / len(all_neg_list),
            'th':thresh.item()
        }
    return test_stat








def train_one_epoch_loc(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, weight=1.0,bi_class_weight=[1,1],s=1, RDrop = 0.):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    TP,FP,TN,FN = 0,0,0,0

    for samples,masks,targets in metric_logger.log_every(
            tqdm(data_loader), print_freq, header):
        samples = samples.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        bs = len(targets)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            outputs,binary_outputs,loc_binary_outputs = model(samples,masks)
            m_loss = criterion(outputs, targets)
            bi_labels = targets.clone()
            bi_labels[bi_labels < 1] = 0
            bi_labels[bi_labels >= 1] = 1
            bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
            if model.module.multi_head:
                # binary_outputs *=s # need to scale up output if binary head norm feature
                probs = F.softmax(binary_outputs.clone().detach(), dim=1)
                cls_loss = F.cross_entropy(binary_outputs, bi_labels, bi_class_weight)
                cls_loc_loss = F.cross_entropy(loc_binary_outputs, bi_labels, bi_class_weight)
            else:
                real_dist = outputs[:, :1]
                fake_dist = outputs[:, 1:]
                cls_loss,probs = call_cls_loss(real_dist,fake_dist,bi_labels,bi_class_weight,s)
                cls_loc_loss = 0

            loss = weight[1]*cls_loss + weight[2]*cls_loc_loss + weight[0] * m_loss
            if RDrop:
                outputs_, binary_outputs_ = model(samples, targets)
                kl_loss_mul = compute_kl_loss(outputs, outputs_)
                loss += RDrop * kl_loss_mul
                kl_loss_bi = compute_kl_loss(binary_outputs, binary_outputs_)
                # print(kl_loss_mul,kl_loss_bi)
                loss += RDrop * kl_loss_bi
        # metric
        loss_value = loss.item()
        acc_multi = accuracy(outputs, targets)
        acc_bi = accuracy(probs,bi_labels)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        P = TP / (TP + FP + 1e-8)
        R = TP / (TP + FN + 1e-8)
        F1 = 2 * P * R / (P + R + 1e-8)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)
        # torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        metric_logger.meters['P'].update(P, n=bs)
        metric_logger.meters['R'].update(R, n=bs)
        metric_logger.meters['f1'].update(F1, n=bs)
        metric_logger.meters['loss_multi'].update(m_loss.item(), n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
        metric_logger.meters['loss_bi_loc'].update(cls_loc_loss.item(), n=bs)
        if RDrop:
            metric_logger.meters['loss_RDrop'].update((kl_loss_mul + kl_loss_bi).item(), n=bs)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_loc(data_loader, model, criterion, device, scale=1,bi_class_weight=[1,1], weight=1.0):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    # switch to evaluation mode
    model.eval()
    pos_list, neg_list,label_list= [], [],[]
    TP,FP,TN,FN = 0,0,0,0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs, binary_outputs,_ = model(images, None)
        m_loss = criterion(outputs, targets)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        bi_class_weight = torch.tensor(bi_class_weight).to(outputs)
        if model.module.multi_head:
            probs = F.softmax(binary_outputs, dim=1)
            cls_loss = F.cross_entropy(binary_outputs,bi_labels,bi_class_weight)
        else:
            real_dist = outputs[:, :1]
            fake_dist = outputs[:, 1:]
            cls_loss, probs = call_cls_loss(real_dist, fake_dist, bi_labels, bi_class_weight, model.module.head.scale)
        # loss = cls_loss + weight * m_loss
        scores = probs[:, 1].tolist()
        for i in range(len(targets)):
            if targets[i] == 0:
                pos_list.append(scores[i])
            else:
                neg_list.append(scores[i])
        acc_bi = accuracy(probs, bi_labels)
        acc_multi = accuracy(outputs,targets)
        pred_labels_bi = probs.argmax(1)
        TP += ((pred_labels_bi == 0) & (bi_labels == 0)).sum().item()
        FP += ((pred_labels_bi == 0) & (bi_labels == 1)).sum().item()
        FN += ((pred_labels_bi == 1) & (bi_labels == 0)).sum().item()
        bs = images.shape[0]
        metric_logger.meters['acc_multi'].update(acc_multi[0].item(), n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi[0].item(), n=bs)
        metric_logger.meters['loss_multi'].update(m_loss.item(), n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
    metric_logger.synchronize_between_processes()
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)
    test_stat['P'] = P
    test_stat['R'] = R
    test_stat["F1"] = F1
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32)).sort()[0]
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32)).sort()[0]
    print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')
    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(
            f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        # print(all_pos_list[int(len(all_pos_list) * (1-threshold_r)) + 1:])
        test_stat[f'tnr_{threshold_r}'] = {
            'TNR':neg_cnt.item() / len(all_neg_list),
            'th':thresh.item()
        }
    return test_stat

def train_one_epoch_mullabel(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, weight=1.0,bi_class_weight=[1,1],s=1, RDrop = 0.,scheduler=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    iter_steps=  len(data_loader)
    idx = 0
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bi_class_weight[1]/bi_class_weight[0]))
    bce = bce.to(device)
    # ls_ce = LabelSmoothingCrossEntropy(smoothing=0.1)
    for samples, targets, multi_labels in metric_logger.log_every(
            tqdm(data_loader), print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        multi_labels = torch.stack([torch.tensor(label, device=device) for label in multi_labels])
        #print(multi_labels)
        bs = len(targets)
        if mixup_fn is not None:
            samples, targets, multi_labels = mixup_fn(samples, targets, multi_labels)
        with torch.cuda.amp.autocast():
            outputs,binary_outputs = model(samples)
            ml_labels = multi_labels.clone()
            ml_labels[ml_labels < 1] = 0
            ml_labels[ml_labels >= 1] = 1
            ml_labels = torch.transpose(ml_labels, 0, 1)
            one_hot = ml_labels
            outputs = outputs.to(torch.float32)
            one_hot = one_hot.to(torch.float32)
            m_loss = criterion(outputs, one_hot)

            bi_labels = targets.clone()
            bi_labels[bi_labels < 1] = 0
            bi_labels[bi_labels >= 1] = 1
            bi_labels = bi_labels.unsqueeze(1).to(outputs)
            cls_loss = bce(binary_outputs, bi_labels)
            loss = cls_loss + weight * m_loss
            if RDrop:
                outputs_, binary_outputs_ = model(samples)
                kl_loss_mul = compute_kl_loss(outputs, outputs_)
                loss += RDrop * kl_loss_mul
                kl_loss_bi = compute_kl_loss(binary_outputs, binary_outputs_)
                # print(kl_loss_mul,kl_loss_bi)
                loss += RDrop * kl_loss_bi
        # metric
        loss_value = loss.item()
        m_probs =  F.sigmoid(outputs)
        mTP = ((m_probs > 0.5 ) & (one_hot == 1)).sum().item()
        mFP = ((m_probs > 0.5) & (one_hot == 0)).sum().item()
        mTN = ((m_probs <= 0.5) & (one_hot == 0)).sum().item()
        mFN = ((m_probs <= 0.5) & (one_hot == 1)).sum().item()
        acc_multi = mTP/bs
        b_probs = F.sigmoid(binary_outputs)
        TP = ((b_probs < 0.5) & (bi_labels == 0)).sum().item()
        FP = ((b_probs < 0.5) & (bi_labels == 1)).sum().item()
        TN = ((b_probs >= 0.5) & (bi_labels == 1)).sum().item()
        FN = ((b_probs >= 0.5) & (bi_labels == 0)).sum().item()
        acc_bi =  (TP+TN)/(TP+TN+FP+FN)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)
        # torch.cuda.synchronize()
        if scheduler:
            scheduler.step(epoch + idx/iter_steps)
        idx+=1
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.meters['acc_multi'].update(acc_multi, n=bs)
        metric_logger.meters['acc_bi'].update(acc_bi, n=bs)
        metric_logger.meters['loss_multi'].update(m_loss.item(),n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
        if RDrop:
            metric_logger.meters['loss_RDrop'].update((kl_loss_mul+kl_loss_bi).item(), n=bs)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_mullabel(data_loader, model, criterion, device, scale=1,bi_class_weight=[1,1], weight=1.0):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    # switch to evaluation mode
    model.eval()
    pos_list, neg_list,label_list= [], [],[]
    TP,FP,TN,FN = 0,0,0,0
    mTP, mFP, mTN, mFN = 0, 0, 0, 0
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bi_class_weight[1]/bi_class_weight[0]))
    bce = bce.to(device)
    total = 0
    for images, targets, multi_labels in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        multi_labels = torch.stack([torch.tensor(label, device=device) for label in multi_labels])
        outputs, binary_outputs = model(images)
        ml_labels = multi_labels.clone()
        ml_labels[ml_labels < 1] = 0
        ml_labels[ml_labels >= 1] = 1
        ml_labels = torch.transpose(ml_labels, 0, 1)
        one_hot = ml_labels
        outputs = outputs.to(torch.float32)
        one_hot = one_hot.to(torch.float32)
        m_loss = criterion(outputs, one_hot)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        bi_labels = bi_labels.unsqueeze(1).to(outputs)
        cls_loss = bce(binary_outputs, bi_labels)
        b_probs = F.sigmoid(binary_outputs)
        scores = b_probs.clone()
        pos_list.extend(scores[targets==0].flatten().tolist())
        neg_list.extend(scores[targets!=0].flatten().tolist())

        m_probs = F.sigmoid(outputs)
        mTP += ((m_probs > 0.5) & (one_hot == 1)).sum().item()
        mFP += ((m_probs > 0.5) & (one_hot == 0)).sum().item()
        mTN += ((m_probs <= 0.5) & (one_hot == 0)).sum().item()
        mFN += ((m_probs <= 0.5) & (one_hot == 1)).sum().item()
        TP += ((b_probs < 0.5) & (bi_labels == 0)).sum().item()
        FP += ((b_probs < 0.5) & (bi_labels == 1)).sum().item()
        TN += ((b_probs >= 0.5) & (bi_labels == 1)).sum().item()
        FN += ((b_probs >= 0.5) & (bi_labels == 0)).sum().item()
        bs = images.shape[0]
        total += bs
        # metric_logger.meters['acc_multi'].update(acc_multi, n=bs)
        # metric_logger.meters['acc_bi'].update(acc_bi.item(), n=bs)
        metric_logger.meters['loss_multi'].update(m_loss.item(), n=bs)
        metric_logger.meters['loss_bi'].update(cls_loss.item(), n=bs)
    metric_logger.synchronize_between_processes()
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # acc_multi = (mTP + mTN) / (mTP + mTN + mFP + mFN)
    acc_multi = mTP/total
    acc_bi = (TP + TN) / (TP + TN + FP + FN)
    test_stat['acc_multi'] = acc_multi
    test_stat['acc_bi'] = acc_bi

    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32))
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32))
    print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')
    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        # print(all_pos_list[int(len(all_pos_list) * (1-threshold_r)) + 1:])
        test_stat[f'tnr_{threshold_r}'] = {
            'TNR':neg_cnt.item() / len(all_neg_list),
            'th':thresh.item()
        }
    print("Averaged stats:", test_stat)
    return test_stat


def evaluate_mullabel_test(data_loader, model, criterion, device, scale=0,bi_class_weight=[1,1], weight=1.0):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    pos_list, neg_list = [], []
    TP,FP,TN,FN = 0,0,0,0
    for images, targets, multi_labels  in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs, binary_outputs = model(images)
        bi_labels = targets.clone()
        bi_labels[bi_labels < 1] = 0
        bi_labels[bi_labels >= 1] = 1
        bi_labels = bi_labels.unsqueeze(1)
        b_probs = F.sigmoid(binary_outputs)
        scores = b_probs.clone()
        pos_list.extend(scores[targets == 0].flatten().tolist())
        neg_list.extend(scores[targets != 0].flatten().tolist())
        TP += ((b_probs < 0.5) & (bi_labels == 0)).sum().item()
        FP += ((b_probs < 0.5) & (bi_labels == 1)).sum().item()
        TN += ((b_probs >= 0.5) & (bi_labels == 1)).sum().item()
        FN += ((b_probs >= 0.5) & (bi_labels == 0)).sum().item()
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    test_stat={
        'acc_bi': (TP + TN) / (TP + TN + FP + FN)
    }
    all_pos_list = torch.from_numpy(np.array(pos_list, dtype=np.float32))
    all_neg_list = torch.from_numpy(np.array(neg_list, dtype=np.float32))
    print(f'pos len: {len(all_pos_list)}, neg len: {len(all_neg_list)}')
    threshold_rs = [0.002, 0.005]
    for threshold_r in threshold_rs:
        threshold = int(len(all_pos_list) * threshold_r) + 1
        thresh = all_pos_list.topk(threshold)[0][-1]
        pos_cnt = torch.sum(all_pos_list > thresh)
        neg_cnt = torch.sum(all_neg_list > thresh)
        print(f'thresh: {thresh:.6f}, pos_cnt: {pos_cnt.item()}, neg_cnt: {neg_cnt.item()}, neg_ratio: {neg_cnt.item() / len(all_neg_list):.6f}')
        test_stat[f'tnr_{threshold_r}'] = {
            'TNR':neg_cnt.item() / len(all_neg_list),
            'th':thresh.item()
        }
    return test_stat
