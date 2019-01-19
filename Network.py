import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

class Net(nn.Module):
    #layers是长度为3的数组，分别是l1，l2，l3的输出维数，l4的输出维数默认为4种type
    def __init__(self,layers,type=5,component=1):
        super(Net, self).__init__()
        # 输入向量的维数为：观测区域数 34 *年数 4 *成分数 5
        self.l1 = nn.Linear(34*4*component+5, layers[0]) #
        self.l2 = nn.Linear(layers[0], layers[1]) # layers[0]:第一个隐藏层的输入向量大小
        self.l3 = nn.Linear(layers[1], layers[2]) # layers[1]:第二个隐藏层的输入向量大小
        self.l4 = nn.Linear(layers[2], type) # layer[2]:第三个隐藏层的输入向量大小
                                             # type为诊断结果类型 5 种，即输出层的向量大小

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # 第四层的输出做softmax处理,这里使用log_softmax和之后的NLLLoss匹配，等价与 softmax+crossentropy
        return F.log_softmax(self.l4(x), dim=1)

    # 参数初始化
    def _initialize_weights(self):
        # print(self.modules())
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                print(m.weight)
