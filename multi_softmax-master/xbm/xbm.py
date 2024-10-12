# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch


class XBM:
    def __init__(self, K,dim):
        self.K = K
        self.feats = torch.zeros(self.K, dim).cuda()
        self.targets = torch.ones(self.K, dtype=torch.long).cuda()*1e-8
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != 1e-8

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size