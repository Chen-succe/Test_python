import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import optimizer
from torch import optim
from demo2_data import cat_and_dog
import glob
from torchvision.models import mobilenet_v2
from torchvision.models import vgg16

path = r'E:\自制数据集\猫狗_all'

# models = mobilenet_v2(pretrained=False)
models = vgg16(pretrained=True)
models.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 100),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(100, 2)
)
# print(model)
# num_class = 2
# models.classifier = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(1280, num_class)
# )

print('hi')


class NN(nn.Module):
    # 初始化神经网络
    def __init__(self, hidden_dim1, hidden_dim2, out_dim):
        super(NN, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=hidden_dim1, kernel_size=(3, 3), stride=(2, 2))
        self.layer1_1 = nn.Conv2d(in_channels=hidden_dim1, out_channels=2*hidden_dim1, kernel_size=(3, 3), padding=1)
        self.layer2 = nn.Conv2d(in_channels=hidden_dim1, out_channels=hidden_dim2, kernel_size=(3, 3), stride=(2, 2))
        self.layer3 = nn.Conv2d(in_channels=hidden_dim2, out_channels=16, kernel_size=(3, 3), stride=(2, 2))

        self.line = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2)
                                  )
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.avg = nn.AdaptiveAvgPool2d((2, 2))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        # x = self.layer1_1(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer2(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer3(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.avg(x)
        # print(x.shape)  # torch.Size([8, 1, 1])
        x = x.contiguous().view(-1, 64)
        x = self.line(x)
        # x = self.relu(x)
        x = self.dropout(x)
        x = nn.functional.softmax(x, dim=-1)

        return x


# 定义计算环境
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # cuda:0
print(device)

# 加载数据
custom_dataset = cat_and_dog(path)
train_size = int(len(custom_dataset) * 0.8)
val_size = int(len(custom_dataset) * 0.2)
train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print('训练集大小：{}'.format(len(train_dataset)))
print('验证集大小：{}'.format(len(val_dataset)))


# 定义一个推理函数，来计算并返回准确率
def infer(model, dataset, device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas, label = data
            outputs = model(datas.to(device))
            # print('result:{}'.format(outputs))
            predict_y = torch.max(outputs, dim=1)[1]
            # print('predict:{}'.format(predict_y))
            # print('label:{}'.format(label))
            acc_num += torch.eq(predict_y, label.to(device)).sum().item()
    acc = acc_num / len(dataset)
    return acc


# 定义主函数，训练
def main(lr=0.01, epoch=20):
    # model = NN(32, 64, 2).to(device)
    model_ = models
    model_ = model_.to(device)
    loss_f = nn.CrossEntropyLoss()

    # pf = [p for p in model.parameters() if p.requires_grad]
    # optimize = optim.Adam(pf, lr=lr)
    optimize = optim.SGD(model_.parameters(), lr, momentum=0.9)

    # 权重文件存储路径
    save_path = os.path.join(os.getcwd(), 'results/weights')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 开始训练
    for epo in range(epoch):
        model_.train()
        acc_num = 0
        sample_num = 0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        for datas in train_bar:
            data, label = datas
            # label = label.squeeze(-1)
            sample_num += data.shape[0]
            optimize.zero_grad()
            outputs = model_(data.to(device))
            pred_class = torch.max(outputs, dim=1)[1]
            # torch.max()返回值是一个数组，第一个元素是max的值，第二元素是max索引的值
            acc_num += torch.eq(pred_class, label.to(device)).sum().item()
            # print('acc_num:{}'.format(acc_num))
            # print('sample_num:{}'.format(sample_num))

            loss = loss_f(outputs, label.to(device))
            loss.backward()
            optimize.step()

            train_acc = acc_num / sample_num
            train_bar.desc = 'train epoch[{}/{}] loss:{:.3f} train_acc:{:.3f}'.format(epo, epoch, loss, train_acc)

        val_acc = infer(model_, val_loader, device)
        print('train epoch[{}/{}] loss:{:.3f} val_acc:{:.3f}'.format(epo, epoch, loss, val_acc))
        torch.save(model_.state_dict(), os.path.join(save_path, 'nn.pth'))
        # 每次数据迭代完，要对初始化的指标清零
        train_acc = 0
        val_acc = 0
    print('Fished Training')

    test_acc = infer(model_, val_loader, device)
    print('test_acc:{}'.format(test_acc))


if __name__ == '__main__':
    main(lr=0.001, epoch=20)
    # path = glob.glob('D:/D/PythonFile/2024_test/Dog_cat_classifier/**.py')
    # print(path)

