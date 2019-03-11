import Network
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import torch
import Dataset
import os 


def config(shape=(100,100,100),classnum=2,learningrate=0.01,learningrateschema=optim.SGD,testdata='',validatedata='',traindata=(),epoch=100,upsamplenum=False,nomalization=None,cnn=False,datapath=False):
    # change work path and make directory
    if datapath:
        os.chdir(datapath)

    # hypeparameters/weights initialize

    print(f'shape:{shape}')
    print(f'classnum:{classnum}')
    print(f'learningrate:{learningrate}')
    #print(f'learningrateschema:{learningrateschema}')
    print(f'testdata:{testdata}')
    print(f'validatedata:{validatedata}')
    print(f'traindata:{traindata}')
    print(f'epoch:{epoch}')
    print(f'upsamplenum:{upsamplenum}')
    print(f'nomalization:{nomalization}')    

    global model
    if cnn:
        model = Network.CNN(classnum)
        print('we created a CNN')
    else: 
        model = Network.Net(shape,classnum)
        print(f'we created a NN')
    model.cuda()
    model._initialize_weights()
    
    # SGD plus momentum
    global optimizer 
    optimizer = learningrateschema(model.parameters(), lr=learningrate, momentum=0.5)#, weight_decay=1e-5)

    # get the dataloader
    global train_loader
    global test_loader
    global validate_loader 
    train_loader, validate_loader, test_loader = Dataset.getloader(upsamplenum,traindata,validatedata,testdata)
   
    for i in range(epoch):
        train(i,nomalization=nomalization)
        validate()
        test()
         
def train(epoch,nomalization=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()
        
        l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0)
        # 将上一批次的梯度计算值置零 set the former batch's gradient value zero
        optimizer.zero_grad()
        output = model(data)
        # lossfunction: cross entropy,we use NLLLoss here（Negative Log Likelihood）
        # cause we take the log of the output tensor before
        loss = F.nll_loss(output, target)
        
        l1_regularization = 0
        l2_regularization = 0
        
        l1lambda = 0.1
        l2lambda = 0.1
  
        if nomalization:  
            for param in model.parameters():
                if nomalization == 'L1':
                    l1_regularization += torch.norm(param, 1)
                elif nomalization == 'L2':
                    l2_regularization += torch.norm(param, 2)
        loss = loss + l1lambda*l1_regularization + l2lambda*l2_regularization

        loss.cuda()
        loss.backward()
        # update weights 
        optimizer.step()

        if batch_idx % 1000 == 0:
            # the output is like：Train Epoch: 1 [0/60000 (0%)]   Loss: 2.292192
            #             Train Epoch: 1 [12800/60000 (21%)]  Loss: 2.289466
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validate():
    validate_loss = 0
    correct = 0
    for data, target in validate_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        # calculate the sum of loss for validate set
        validate_loss += F.nll_loss(output, target).data.item()
        # max means the prediction
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validate_loss /= len(validate_loader.dataset)
    # the output is like0m~ZValidate set: Average loss: 0.0163, Accuracy: 6698/10000 (67%)
    print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validate_loss, correct, len(validate_loader.dataset),
        100. * correct / len(validate_loader.dataset)))

def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        # calculate the sum of loss for testset
        test_loss += F.nll_loss(output, target).data.item()
        # max means the prediction
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    # the output is like：Test set: Average loss: 0.0163, Accuracy: 6698/10000 (67%)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    #config(shape=(100,100,100),classnum=5,learningrate=0.001,learningrateschema=optim.SGD,testdata='testdata.csv',validatedata='validatedata.csv',traindata=('0.csv','1.csv','2.csv','3.csv','4.csv'),epoch=100,upsamplenum=100000,nomalization='L1')
    #config(shape=(50,50,50),classnum=5,learningrate=0.01,learningrateschema=optim.SGD,testdata='testdata.csv',validatedata='validatedata.csv',traindata=('0.csv','1.csv','2.csv','3.csv','4.csv'),epoch=100,upsamplenum=100000,nomalization=False,cnn=True,datapath='/data/dataaugmentationinmedicalfield')
    config(shape=(100,100,100),classnum=5,learningrate=0.01,learningrateschema=optim.SGD,testdata='testdata.csv',validatedata='validatedata.csv',traindata=('0.csv','1.csv','2.csv','3.csv','4.csv'),epoch=100,upsamplenum=False,nomalization=False,cnn=False,datapath='/data/dataaugmentationinmedicalfield')
