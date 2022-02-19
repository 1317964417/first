++++++++++++++++++++++++++++++++++今天是2022年2月15号 ++++++++++++++++++++++++++++++++++
就目前为止，代码仍有问题存在
1、首先CW和PGD的训练时间过长，没有保存模型如果感兴趣可以运行这两个文件以获取对应的网络模型.pkl
2、代码任然需要修改，目前是最原始的版本 

如何运行代码 
1、首先，需要安装相对应的环境 torch和torchvision最新版本就行
2、另外，本人本着能偷懒就偷懒的想法，FGSM和PGD以及CW等都是从cleverhans工具包(不仅仅支持tf也有支持torch的)里面调用的，所以想要运行还需要安装这个工具包，具体是pip install cleverhans 
ps:     from absl import app, flags
        from easydict import EasyDict
        import numpy as np
        import torch as t
        import torchvision as tv
        import torchvision.transforms as transforms
        from torchvision.transforms import ToPILImage
        from torch.autograd import Variable
        from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
        from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
有以上这些玩意即可
3、正式进入正题
 3.1、首先运行model1.py文件 切记路径，需要添加 root='./cifar-10-python/',
 3.2、其次再运行data_process.py文件 这样可以将数据集解压，然后按照训练集和测试集分类
 3.3、运行NBC_new.py文件 该代码用来提取每个神经元的最大输出与最小输出
 3.4、运行FGSM获得受攻击后的模型cifar10_3_adv_FGSM.pkl
 3.5、运行NBC_adv_PGD_and_FGSM.py，目前就只有cifar10_3_adv_FGSM.pkl这一个模型可以用，然后获取最大最小输出
  ![image](https://github.com/1317964417/Test/blob/main/picture/%E5%9B%BE%E7%89%872.png)
  ![image](https://github.com/1317964417/Test/blob/main/picture/%E5%9B%BE%E7%89%873.png)
  ![image](https://github.com/1317964417/Test/blob/main/picture/%E5%9B%BE%E7%89%871.png)
 
 后面的继续更新


