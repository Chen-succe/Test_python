import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms


class cat_and_dog(Dataset):
    def __init__(self, root_path):
        self.path = root_path
        self.data, label = self.get_image()
        self.data_len = len(label)
        self.label = torch.from_numpy(np.array(label, dtype='int64'))

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        self.data = list(self.data)
        self.label = list(self.label)
        image_data = self.data[item]
        image_data = self.get_image_pix(image_data)

        return image_data, self.label[item]

    def get_image(self):
        path = r'{}'.format(self.path)
        images_path = []
        labels = []
        for dir_root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jpg'):
                    im_path = os.path.join(dir_root, file)
                    if os.path.exists(im_path):
                        if 'dog' in im_path:
                            label = 0
                        elif 'cat' in im_path:
                            label = 1
                        else:
                            raise Warning('wrong')
                        images_path.append(im_path)
                        labels.append(label)
        return images_path, labels

    def get_image_pix(self, pt):
        pt = r'{}'.format(pt)
        image = Image.open(pt)
        # image = np.asarray(image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(),
            # transforms.RandomCrop(50),
            # transforms.RandomResizedCrop(150),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image_tensor = transform(image)
        # image_tensor = torch.unsqueeze(image_tensor, 0)
        # print(image_tensor.shape)  # torch.Size([1, 3, 244, 224])
        return image_tensor


'''
要自定义我们自己的数据集类，需要重写数据加载类，继承Dataset类，继承这个类必须要实现
上面三个函数。
'''

if __name__ == '__main__':
    path = r'E:\自制数据集\猫狗_test'
    path = r''
