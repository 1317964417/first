import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms
import torchvision.transforms as transforms
import cv2
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
ptt = torchvision.transforms.ToTensor()
show = ToPILImage()  # 可以把tensor转为image
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 转为Tensor，把灰度范围从0-255变换到0-1，归一化
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 把0-1变为-1到1，标准化
# ])
# # 测试集
# testset = tv.datasets.CIFAR10(
#     root='./cifar-10-python/',
#     train=False,
#     download=True,
#     transform=transform)
#
# testloader = torch.utils.data.DataLoader(
#     testset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=0)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

class Net(nn.Module):
    # 把网络中具有可学习参数的层放在构造函数__inif__中
    def __init__(self):
        # 下式等价于nn.Module.__init__.(self)
        super(Net, self).__init__()  # RGB 3*32*32
        self.conv1 = nn.Conv2d(3, 15, 3)  # 输入3通道，输出15通道，卷积核为3*3
        self.conv2 = nn.Conv2d(15, 75, 4)  # 输入15通道，输出75通道，卷积核为4*4
        self.conv3 = nn.Conv2d(75, 375, 3)  # 输入75通道，输出375通道，卷积核为3*3
        self.fc1 = nn.Linear(1500, 400)  # 输入2000，输出400
        self.fc2 = nn.Linear(400, 120)  # 输入400，输出120
        self.fc3 = nn.Linear(120, 84)  # 输入120，输出84
        self.fc4 = nn.Linear(84, 10)  # 输入 84，输出 10（分10类）

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 75*6*6   -> 375*4*4   -> 375*2*2
        x = x.view(x.size()[0], -1)  # 将375*2*2的tensor打平成1维，1500
        x = F.relu(self.fc1(x))  # 全连接层 1500 -> 400
        x = F.relu(self.fc2(x))  # 全连接层 400 -> 120
        x = F.relu(self.fc3(x))  # 全连接层 120 -> 84
        x = self.fc4(x)  # 全连接层 84  -> 10
        return x

model = Net()
print(model.eval())
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
path = "./cifar10_3.pkl"
model = torch.load(path)
model.fc4.register_forward_hook(get_activation('fc4'))

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# print(images.size())
# outputs = model(Variable(images))
path_of_picture_of_airplane  =  './cifar-10-python/cifar-10-batches-py/test/airplane/'
counter = 0
list = []
for root, dirs, files in os.walk(path_of_picture_of_airplane):
    for file in files:
        path = os.path.join(root, file)
        picture = cv2.imread(path)
        # cv2.imshow("p",picture)
        picture = ptt(picture)
        picture = picture.unsqueeze(0)
        outputs = model(Variable(picture))
        # print(activation['fc4'])
        outputs = outputs.detach().numpy()
        list.append(outputs[0][0])  # 最后一层输出的一个神经元
        # cv2.waitKey(10000)
        counter += 1
        if counter == 1000:
            break
print(np.max(list),np.min(list),len(list))



'''
dataiter = iter(testloader)  # 采用iter(dataloader)返回的是一个迭代器，然后可以使用next()访问。iter(dataloader)访问时，
# imgs在前，labels在后，分别表示：图像转换0~1之间的值，labels为标签值。并且imgs和labels是按批次进行输入的。
images, labels = dataiter.next()
print("实际的label:", " ".join("%08s" % classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100))
# for name, m in model.named_modules():
#     print(m,name)
# 计算图片在每个类别上的分数
outputs = model(Variable(images))

# 得分最高的那个类
_, predicted = torch.max(outputs.data, 1)  # torch.max()返回两个值，第一个值是具体的value，，也就是输出的最大值（我们用下划线_表示 ，指概率），
# 第二个值是value所在的index（也就是predicted ， 指类别）
# 选用下划线代表不需要用到的变量
# 数字1：其实可以写为dim=1，表示输出所在行的最大值，若改写成dim=0则输出所在列的最大值
print("预测结果:", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

Files already downloaded and verified
Net1(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
实际的label:      cat     ship     ship    plane
预测结果:  frog plane plane plane
'''

