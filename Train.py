import Network
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import torch
import Dataset
import os
import pickle

def config(shape=[100,100,100],classnum=2,binaryafter=False,learningrate=0.01,learningrateschema=optim.SGD,batchsize=64,testdata='',validatedata='',traindata=(),epoch=100,samplenum=False,sampletype=False,l1regularization=None,l2regularization=None,cnn=False,datapath=False,batchnorm=False,dropout=False):
    
    # binary or muti classification set

    traindata = []

    if classnum > 2:

        testdata = 'testdata-muti.csv'
        validatedata = 'validatedata-muti.csv'

        for i in range(classnum):
            traindata.append(f'{i}-muti.csv')

    elif classnum == 2:

        testdata = 'testdata-binary.csv'
        validatedata = 'validatedata-binary.csv'

        for i in range(classnum):
            traindata.append(f'{i}-binary.csv')
    
    # print the configuaration
    print(f'latent-layer-shape:{shape}')
    print(f'the-num-of-classes:{classnum}')
    print(f'learningrate:{learningrate}')
    #print(f'learningrateschema:{learningrateschema}')
    print(f'batchsize:{batchsize}')
    print(f'testdata:{testdata}')
    print(f'validatedata:{validatedata}')
    print(f'traindata:{traindata}')
    print(f'epoch:{epoch}')
    print(f'samplenum:{samplenum}')
    print(f'sampletype:{sampletype}')
    print(f'l1regularizationrate:{l1regularization}')
    print(f'l2regularizationrate:{l2regularization}')
    print(f'batchnormmomentom:{batchnorm}')
    print(f'dropoutrate:{dropout}')
    print(f'cnn:{cnn}')
    print(f'path:{datapath}')
       

    global model
    
    if datapath:
        os.chdir(datapath)    
    testdata = pd.read_csv(testdata)

    if binaryafter:
        classnum = 2
        testdata.loc[testdata['label'] > 1, 'label'] = 1
   
    model = Network.Net(shape,classnum,batchnorm=batchnorm,dropout=dropout,cnn=cnn)
        
    model.cuda()
    model._initialize_weights()
    
    # SGD plus momentum
    global optimizer 
    optimizer = learningrateschema(model.parameters(), lr=learningrate, momentum=0.5)#, weight_decay=1e-5)

    # get the dataloader
    global train_loader
    global test_loader
    global validate_loader
    global relprop_loader

    train_loader, validate_loader, test_loader = Dataset.getloader(samplenum,sampletype,batchsize,traindata,validatedata,testdata,classnum,binaryafter,datapath)
    
    for feature in ['sex','visit_age','scanner']:
        if feature in testdata.columns:
            testdata.drop(columns=feature,inplace=True)

    relprop_loader = torch.utils.data.DataLoader(dataset=Dataset.MyDataset(testdata),
                                              batch_size=1,
                                              shuffle=False)
    
    accuracy = []
 
    for i in range(epoch):
        train(i,l1regularization=l1regularization,l2regularization=l2regularization)
        validate()
        accuracy.append(test().cpu().numpy())
    
    accuracy = pd.DataFrame(accuracy).T
    print(accuracy)
    print(f'median:{accuracy.median} max:{accuracy.max()}')
    
    # relevance score computation
    relprop()
         
def train(epoch,l1regularization=None,l2regularization=None):
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()
        
        l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0)
        # set the former batch's gradient value zero
        optimizer.zero_grad()
        output = model(data)
        # lossfunction: cross entropy,we use NLLLoss here, Negative Log Likelihood
        # cause we take the log of the output tensor before
        loss = F.nll_loss(output, target)
        
        l1_regularization = 0
        l2_regularization = 0
        
        l1lambda = l1regularization
        l2lambda = l2regularization
  
        if l1regularization or l2regularization:  
            for param in model.parameters():
                if l1regularization:
                    l1_regularization += torch.norm(param, 1)
                if l2regularization:
                    l2_regularization += torch.norm(param, 2)
        loss = loss + l1lambda*l1_regularization + l2lambda*l2_regularization

        loss.cuda()
        loss.backward()
        # update weights 
        optimizer.step()

        if batch_idx % 1000 == 0:
            # the output is like, Train Epoch: 1 [0/60000 (0%)]   Loss: 2.292192
            #             Train Epoch: 1 [12800/60000 (21%)]  Loss: 2.289466
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validate():
    
    model.eval()
    
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
    
    model.eval()
    
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
    # the output is like Test set: Average loss: 0.0163, Accuracy: 6698/10000 (67%)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # record the correct predict type
    '''pred = pred.cpu().numpy()
    pred = pd.DateFrame(pred)
    target.data.cpu().numpy()
    target = pd.DataFrame(target)
    correctpred = pred.loc[pred == target]
    print(correctpred.count())'''

    return 100. * correct / len(test_loader.dataset)

def relprop():

    model.eval()

    relevance_scores = []

    for data, target in relprop_loader:
        
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        
        # max means the prediction

        pred = output.data.max(1, keepdim=True)[1][0]

        if pred.eq(target.data):
            print('one correct') 
            relevance_score = {}
            relevance_score['data'] = data.cpu().numpy()
            relevance_score['label'] = target.cpu().numpy()
            relevance_score['score'] = model.relprop()
            relevance_scores.append(relevance_score)
        
    # data persistence
    with open('relevance_scores.pk', 'wb+') as f:
        pickle.dump(relevance_scores, f)
    
    
if __name__ == '__main__':
    config(shape=[50,50,50,50],
           classnum=5,
           binaryafter=True,
           learningrate=0.001,
           learningrateschema=optim.SGD,
           batchsize=128,
           epoch=10,
           samplenum=False,
           sampletype=False,
           l1regularization=False,
           l2regularization=False,
           cnn = False,#[[1,1,2,1,0]],
           datapath='/data/dataaugmentationinmedicalfield/kernal_comb1/',
           batchnorm=0.1,
           dropout=False)
