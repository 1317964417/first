import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim
from torch.autograd import Variable

# 指定參數
kernel_size = (5,5)
batch_size = 4
Epoches =8
lr = 0.01
momentum= 0.9
# 導入數據集
show = ToPILImage() # 可以把Tensor轉換成Image，方便可視化
# 第一次運行程序torchvision會自動下載cifar-10數據集
# 大約有100M左右
# 如果已經下載好了cifar-10，可通過root參數指定

# 定義數據集的處理
transform = transforms.Compose([
    transforms.ToTensor(), # 轉換為Tensor類型
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 歸一化
])

# 訓練集
trainset = torchvision.datasets.CIFAR10(
    root='./cifar-10-python/',
    train= True,
    download=True,
    transform=transform
)
trianloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

# 測試集
testset = torchvision.datasets.CIFAR10(
    root='./cifar-10-python/',
    train = False,
    download=True,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(trainset)
print(testset)
# 定義一個簡單的神經網絡
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x1 = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x2 = x1.view(x1.size()[0], -1)
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = self.fc3(x4)
        return x5
# 打印神經網絡的結構
net=Net1()
print(net)
# 訓練神經網絡
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
torch.set_num_threads(6)
def train():
    for epoch in range(8):
        running_loss = 0.0
        for i, data in enumerate(trianloader, 0):
            # 輸入數據
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清零
            optimizer.zero_grad()
            # forward+backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 更新參數
            optimizer.step()
            # 打印log信息
            # loss是一個scalar，需要使用一個loss.item()來獲取數值，不能使用loss[0]
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
                # 每一千次把當前梯度數值清零
                running_loss = 0.0
    # 保存整個網絡
    torch.save(net, "cifar10_2.pkl")
    # 保存網絡當前的狀態
    torch.save(net.state_dict(), "cifar10_new_2.pkl")
    print("結束訓練")
train()
# 預測正確的圖片數
correct = 0
# 總共的圖片數
total = 0
net.eval()
# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        score = net(Variable(images))
        _, predicted = torch.max(score.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))