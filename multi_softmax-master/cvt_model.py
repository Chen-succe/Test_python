import torch
from timm.models import create_model
from models import *


model = create_model(
        'efficientformerv2_s2',
        num_classes=2,
        distillation=False,
        pretrained=None
    )

ckpt = torch.load('weight/checkpoint.pth')
model.load_state_dict(ckpt['model'])
torch.save(model.state_dict(), 'weight/efformerv2_s2_0217_ep72.pth')


