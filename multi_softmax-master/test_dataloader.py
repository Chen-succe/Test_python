from util import *

import torch
import numpy as np
import random
from torch.backends import cudnn
import torch.utils.data


def fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True


def worker_init_fn(worker_id):
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    path = './huokai_train_PIL_resize224.txt'
    fix_random(42)
    dataset_val = CustomDataset(path, type='train', transforms=get_transform('train', resize=224))
    dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=2, num_workers=2, shuffle=False,
                                             worker_init_fn=worker_init_fn)
    for epoch in range(5):
        print("Epoch:", epoch)
        for idx, (image, labels) in enumerate(dataloader):
            print(f"Label : {labels}")
            print(image[0])
            if idx == 1:
                break
