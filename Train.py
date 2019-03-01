import Network
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import Dataset

def config(shape=(100,100,100),learningrate=0.01,learningrateschema=optim.SGD,testdata='testdata.csv',traindata=()):
    # hypeparameters/weights initialize
    model = Network.Net(shape)
    model.cuda()
    model._initialize_weights()

    # SGD plus momentum
    optimizer = learningrateschema(model.parameters(), lr=learningrate, momentum=0.5)

    # get the dataloader
    train_loader,test_loader = Dataset.getloader(100000,testdata,traindata) 
    # 'data0aug.csv','data1aug.csv','data2aug.csv','data3aug.csv','data4aug.csv')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()
        # 将上一批次的梯度计算值置零 set the former batch's gradient value zero
        optimizer.zero_grad()
        output = model(data)
        # lossfunction: cross entropy,we use NLLLoss here（Negative Log Likelihood）
        # cause we take the log of the output tensor before
        loss = F.nll_loss(output, target)
        loss.cuda()
        loss.backward()
        # update weights 
        optimizer.step()

        if batch_idx % 1000 == 0:
            # the output is like：Train Epoch: 1 [0/60000 (0%)]   Loss: 2.292192
            #             Train Epoch: 1 [12800/60000 (21%)]  Loss: 2.289466
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        # calculate the sum of loss for testset
        test_loss += F.nll_loss(output, target).data[0]
        # max means the prediction
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    # the output is like：Test set: Average loss: 0.0163, Accuracy: 6698/10000 (67%)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    config(shape=(100,100,100),learningrate=0.01,learningrateschema=optim.SGD,testdata='testdata.csv',traindata=('0.csv','1.csv'))
    for epoch in range(100):
        train(epoch)
        test()
