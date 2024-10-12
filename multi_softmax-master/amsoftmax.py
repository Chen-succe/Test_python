import torch
import torch.nn as nn


class AMSoftmax(nn.Module):
    def __init__(self,
                 m=0.3,
                 s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, costh, lb):
        assert costh.size()[0] == lb.size()[0]
        lb_view = lb.view(-1, 1)
        delt_costh = torch.zeros(costh.size()).cuda().scatter_(1, lb_view, self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss


if __name__ == '__main__':
    criteria = AMSoftmax()
    costh = torch.randn(4, 2)
    lb = torch.randint(0, 2, (4, ), dtype=torch.long)
    loss = criteria(costh, lb)

    print(loss.detach().numpy())