"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class ADM_Softmax(Module):
    """Implementation for "Additive Margin Softmax for Face Verification"
    """
    def __init__(self, feat_dim, num_class, margin_r=0.4,margin_f=0.1, scale=5):
        super(ADM_Softmax, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_r = margin_r
        self.margin_f = margin_f
        self.scale = scale
    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        output = cos_theta * 1.0
        if self.training:
            for idx in range(len(labels)):
                l = labels[idx]
                output[idx][l] -= self.margin_r if l ==0 else self.margin_f
        output *= self.scale
        return output

if __name__ == '__main__':
    f = torch.randn(4,8)
    label = [1,2,5,0]
    adm = ADM_Softmax(8,6,scale=5)
    adm(f,label)