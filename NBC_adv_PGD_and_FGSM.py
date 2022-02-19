import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
Show = ToPILImage()
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

ptt = torchvision.transforms.ToTensor()
model = Net()
print(model.eval())
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

path = './cifar10_3_adv_FGSM.pkl'
model = torch.load(path)
model.fc3.register_forward_hook(get_activation('fc3'))
path_of_picture_of_airplane = './cifar-10-python/cifar-10-batches-py/test/airplane/'

counter = 0
list = []
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []
list9 = []

for root, dirs, files in os.walk(path_of_picture_of_airplane):
    for file in files:
        path = os.path.join(root, file)
        picture = cv2.imread(path)
        # cv2.imshow("p",picture)
        picture = ptt(picture)
        picture = picture.unsqueeze(0)
        outputs = model(Variable(picture))
        # print(activation['fc3'])
        outputs = outputs.detach().numpy()
        list.append(outputs[0][0])  # 最后一层输出的一个神经元，毕竟最后一层有10个神经元。
        list1.append(outputs[0][1])
        list2.append(outputs[0][2])
        list3.append(outputs[0][3])
        list4.append(outputs[0][4])
        list5.append(outputs[0][5])
        list6.append(outputs[0][6])
        list7.append(outputs[0][7])
        list8.append(outputs[0][8])
        list9.append(outputs[0][9])
        # cv2.waitKey(10000)
        counter += 1
        if counter == 1000:
            break
print(np.max(list),np.min(list),len(list))
print(np.max(list1),np.min(list1),len(list1))
print(np.max(list2),np.min(list2),len(list2))
print(np.max(list3),np.min(list3),len(list3))
print(np.max(list4),np.min(list4),len(list4))
print(np.max(list5),np.min(list5),len(list5))
print(np.max(list6),np.min(list6),len(list6))
print(np.max(list7),np.min(list7),len(list7))
print(np.max(list8),np.min(list8),len(list8))
print(np.max(list9),np.min(list9),len(list9))




