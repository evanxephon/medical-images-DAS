import Network
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import Dataset

# 模型实例
model = Network.Net([100,100,100])
model.cuda()
model._initialize_weights()

# 采用随机梯度下降来更新权重，并且使用了动量这一参数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 得到dataloader实例
train_loader,test_loader = Dataset.getloader(100000,'testdata.csv','0.csv','1.csv') 
# 'data0aug.csv','data1aug.csv','data2aug.csv','data3aug.csv','data4aug.csv')


# 训练
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()
        # 将上一批次的梯度计算值
        optimizer.zero_grad()
        output = model(data)
        # 损失函数采用了交叉熵 计算采用了NLLLoss-用于多分类的负对数似然损失函数（Negative Log Likelihood）
        # 因为输出向量已经做过对数log操作
        loss = F.nll_loss(output, target)
        loss.cuda()
        loss.backward()
        # 更新参数
        optimizer.step()

        if batch_idx % 1000 == 0:
            # 输出结果如：Train Epoch: 1 [0/60000 (0%)]   Loss: 2.292192
            #             Train Epoch: 1 [12800/60000 (21%)]  Loss: 2.289466
            #                        轮数-批次-完整数据大小--------损失函数
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    # 初始化测试损失函数值和正确预测数
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        # 计算验证集损失函数值
        test_loss += F.nll_loss(output, target).data[0]
        # 得到输出向量中最大的值，那一位也就对应着预测结果
        pred = output.data.max(1, keepdim=True)[1]
        # 计算正确预测数
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    # 输出格式为：Test set: Average loss: 0.0163, Accuracy: 6698/10000 (67%)
    #                                   损失函数值-准确率的计算-准确率
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test()
