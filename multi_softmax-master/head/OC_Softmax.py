"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class OC_Softmax(Module):
    """Implementation for "Additive Margin Softmax for Face Verification"
    """
    def __init__(self, feat_dim, margin_r=0.9,margin_f=0.2, scale=20):
        super(OC_Softmax, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, 1))
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
            output[labels==0] = self.margin_r - output[labels ==0]
            output[labels==1] = output[labels==1] - self.margin_f
            output *= self.scale
        return output

if __name__ == '__main__':
    f = torch.randn(4,8)
    label = np.array([1,2,5,0])
    adm = OC_Softmax(8,scale=5)
    adm(f,label)