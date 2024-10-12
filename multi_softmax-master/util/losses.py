# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
import torch.nn as nn

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class BatchMMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5,margin = 1.0):
        super(BatchMMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
    def guassian_kernel(self, source, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])
        total0 = source.unsqueeze(0).expand(int(source.size(0)), int(source.size(0)), int(source.size(1)))
        total1 = source.unsqueeze(1).expand(int(source.size(0)), int(source.size(0)), int(source.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    def forward(self, source,label,bi_label=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        label = torch.repeat_interleave(label.view([1, -1]), batch_size, dim=0)
        equal =(label==label.T).int()
        unequal = 1 - equal
        equal[torch.eye(batch_size,batch_size).bool()] = 0
        XX = equal * kernels
        XY = unequal * kernels
        """MAX """
        sim = torch.sum(XX,dim=1)/((XX != 0).sum(dim=1) +1e-8) # multi
        un_sim = torch.max(XY,dim=1)[0] # multi
        y = torch.ones_like(sim)
        # loss = un_sim-sim
        # """ margin ，这样margin是不是有问题"""
        loss = self.margin_loss(sim,un_sim,y)
        """new sum"""
        # sim = torch.sum(torch.where(XX!=0,self.kernel_num-XX,XX), dim=1)/((XX!=0).sum(1)+1e-8)
        # un_sim = torch.sum(XY, dim=1)/((XY!=0).sum(1)+1e-8)
        # loss = sim + un_sim
        if bi_label is not None:
            bl0 = bi_label.unsqueeze(0).expand(batch_size,batch_size)
            bl1 = bi_label.unsqueeze(1).expand(batch_size,batch_size)
            bi_equal = (bl0 == bl1).int()
            bi_unequal = 1 - bi_equal
            bi_XY = bi_unequal * kernels
            bi_un_sim = torch.max(bi_XY, dim=1)[0]  # binary
            loss += bi_un_sim
        # un_sim = torch.sum(XY)/((XY != 0).sum() +1e-8)
        return loss.mean()

class XBatchMMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5,margin = 1.0):
        super(BatchMMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.centers = torch.randn(self.num_classes, self.feat_dim)
    def guassian_kernel(self, source,target,kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]+target.size()[0])
        total0 = source.unsqueeze(0).expand(int(source.size(0)), int(source.size(0)), int(source.size(1)))
        total1 = source.unsqueeze(1).expand(int(source.size(0)), int(source.size(0)), int(source.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    def forward(self, source,label,bi_label=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        label = torch.repeat_interleave(label.view([1, -1]), batch_size, dim=0)
        equal =(label==label.T).int()
        unequal = 1- equal
        equal[torch.eye(batch_size,batch_size).bool()] = 0
        XX = equal * kernels
        XY = unequal * kernels
        """MAX"""
        # sim = torch.sum(XX,dim=1)/((XX != 0).sum(dim=1) +1e-8) # multi
        # un_sim = torch.max(XY,dim=1)[0] # multi
        # y = torch.ones_like(sim)
        # loss = un_sim-sim
        """ margin """
        # loss = self.margin_loss(sim,un_sim,y)
        """new sum"""
        sim = torch.sum(torch.where(XX!=0,self.kernel_num-XX,XX), dim=1)
        un_sim = torch.sum(XY, dim=1)
        loss = sim + un_sim

        if bi_label is not None:
            bl0 = bi_label.unsqueeze(0).expand(batch_size,batch_size)
            bl1 = bi_label.unsqueeze(1).expand(batch_size,batch_size)
            bi_equal = (bl0 == bl1).int()
            bi_unequal = 1 - bi_equal
            bi_XY = bi_unequal * kernels
            bi_un_sim = torch.max(bi_XY, dim=1)[0]  # binary
            loss += bi_un_sim
        # un_sim = torch.sum(XY)/((XY != 0).sum() +1e-8)
        return loss.mean()
class Inner_loss(nn.Module):
    def __init__(self,p=1,w=1):
        super(Inner_loss, self).__init__()
        self.p = p
        self.w = w
        self.m = 0
    def forward(self, feature, target):
        bs = int(feature.size()[0])
        feature = F.normalize(feature,dim=1)
        gram_feature = feature@feature.T
        label = target.clone()
        label = torch.repeat_interleave(label.view([1, -1]), bs, dim=0)
        equal = torch.triu((label==label.T).int())
        unequal = torch.triu(1-equal)
        equal[torch.eye(bs,bs).bool()]=0
        sim = equal*gram_feature
        un_sim = unequal * gram_feature
        sim[sim==self.p] = 0
        un_sim[un_sim==-self.p]= 0
        sim = torch.sum(sim,1)
        un_sim = torch.sum(un_sim,1)
        # print(sim.shape)
        # print(un_sim.shape)
        loss = un_sim-self.w*sim
        # print(loss.shape)
        # print(loss)
        return loss.mean()

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
        #           torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = (x * x).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  (self.centers.to(x.dtype) * self.centers.to(x.dtype)).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = distmat - 2*(x@self.centers.to(x.dtype).t())
        # distmat.addmm_(x, self.centers.to(x.dtype).t(),beta=1, alpha = -2 )

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.to(distmat.dtype)
        dist[dist<1e-12] = 1e-12
        dist[dist>1e+12] = 1e+12
        loss = dist.sum()/batch_size
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self,margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, inputs_col, targets_col, inputs_row, target_row):
        n = inputs_col.size(0)
        m = inputs_row.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        # epsilon = 1e-5
        # loss = list()
        # neg_count = list()
        l1 = targets_col.unsqueeze(1).expand(n,m)
        l2 = target_row.unsqueeze(0).expand(n,m)
        equal = (l1 == l2).int()
        unequal = 1 - equal


        sim_mat = torch.where(sim_mat!=1,sim_mat,torch.zeros_like(sim_mat))
        pos_pair  = equal * sim_mat
        neg_pair = unequal *sim_mat

        pos_pair = torch.where(pos_pair!=0,1-pos_pair,pos_pair)
        neg_pair = torch.where(neg_pair > self.margin,neg_pair,torch.zeros_like(neg_pair))

        pos_loss = torch.sum(pos_pair, dim=1)/((pos_pair!=0).sum(1)+1e-8)
        neg_loss = torch.sum(neg_pair, dim=1)/((neg_pair!=0).sum(1)+1e-8)
        loss = pos_loss + neg_loss

        # for i in range(n):
        #     pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
        #     pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
        #     neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)
        #     neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)
        #     # pos_loss = torch.sum(-pos_pair_ + 1)
        #     if len(pos_pair_) > 0:
        #         pos_loss = torch.mean(-pos_pair_ + 1)
        #     else:
        #         pos_loss = 0
        #     if len(neg_pair) > 0:
        #         # neg_loss = torch.sum(neg_pair)
        #         neg_loss = torch.mean(neg_pair)
        #         neg_count.append(len(neg_pair))
        #     else:
        #         neg_loss = 0
        #     loss.append(pos_loss + neg_loss)
        #     # if pos_loss.isnan() or pos_loss.isinf():
        #     #     print(pos_pair_)
        #     #     print(sim_mat[i])
        #     #     print(pos_loss,neg_loss)
        # loss = sum(loss) / n  # / all_targets.shape[1]
        return loss.mean()
