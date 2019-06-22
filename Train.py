import Network
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import torch
import Dataset
import os
import pickle

def config(shape=[100,100,100],classnum=2,classnums=False,binaryafter=False,learningrate=0.01,learningrateschema=optim.SGD,
           batchsize=64,testdata='',validatedata='',traindata=(),epoch=100,samplenum=False,sampletype=False,l1regularization=None,
           l2regularization=None,cnn=False,datapath=False,batchnorm=False,dropout=False,rawdatatrain=False,cvmodeoutput=False,cvnum=0):
    
    # binary or multi classification set
    traindata = []
    
    # specify the chosen classes
    if classnums:
        testdata = 'testdata-multi.csv'
        validatedata = 'validatedata-multi.csv'
        for i in classnums:
            traindata.append(f'{i}-multi.csv')
    # using all the classes
    elif classnum > 2:
        testdata = 'testdata-multi.csv'
        validatedata = 'validatedata-multi.csv'

        for i in range(classnum):
            traindata.append(f'{i}-multi.csv')

    elif classnum == 2:
        testdata = 'testdata-binary.csv'
        validatedata = 'validatedata-binary.csv'

        for i in range(classnum):
            traindata.append(f'{i}-binary.csv')
    if cvnum == 1:
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
       
    if datapath:
        os.chdir(datapath)
    
    model = Network.Net(shape,classnum,batchnorm=batchnorm,dropout=dropout,cnn=cnn)
        
    model.cuda()
    model._initialize_weights()
    
    # SGD plus momentum
    optimizer = learningrateschema(model.parameters(), lr=learningrate, momentum=0.2)#, weight_decay=1e-5)

    # get the dataloader
    train_loader, validate_loader, test_loader, relprop_loader = Dataset.getloader(samplenum=samplenum,sampletype=sampletype,
                                                                                   batchsize=batchsize,traindata=traindata,
                                                                                   validatedata=validatedata,testdata=testdata,
                                                                                   classnum=classnum,classnums=classnums,
                                                                                   binaryafter=binaryafter,datapath=datapath, 
                                                                                   rawdatatrain=rawdatatrain)
    accuracy = []
    valiacc = []
 
    for i in range(epoch):
        train(model, train_loader, optimizer, i,l1regularization=l1regularization,l2regularization=l2regularization,
              cvmodeoutput=cvmodeoutput)
        valiacc.append(validate(model, validate_loader, cvmodeoutput=cvmodeoutput).cpu().numpy())
        accuracy.append(test(model, test_loader, classnum, cvmodeoutput=cvmodeoutput).cpu().numpy())
    
    acc = pd.DataFrame([accuracy, valiacc])

    print(acc)
    
    # relevance score computation
    relprop(model, relprop_loader)
    
    # restore the model 
    torch.save(model.state_dict(), os.path.join(datapath, 'weights.pkl'))
         
def train(model, train_loader, optimizer, epoch, l1regularization=None, l2regularization=None, cvmodeoutput=False):
    
    model.train()
  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()
        
        l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0)
        # set the former batch's gradient value zeroi

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
        if not cvmodeoutput:
            if batch_idx % 1000 == 0:
                # the output is like, Train Epoch: 1 [0/60000 (0%)]   Loss: 2.292192
                #             Train Epoch: 1 [12800/60000 (21%)]  Loss: 2.289466
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def validate(model, validate_loader, cvmodeoutput=False):
    
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
    if not cvmodeoutput:
        validate_loss /= len(validate_loader.dataset)
        # the output is like0m~ZValidate set: Average loss: 0.0163, Accuracy: 6698/10000 (67%)
        print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validate_loss, correct, len(validate_loader.dataset),
            100. * correct / len(validate_loader.dataset)))
    return  100. * correct / len(validate_loader.dataset)

def test(model, test_loader, classnum=5, cvmodeoutput=False):
    
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0

    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))

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
  
        correct += pred.eq(target.data.view_as(pred)).sum()
        
        _, predicted = torch.max(output.data, 1)
        
        pre_mask = torch.zeros(output.size()).scatter_(1, predicted.view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)

    if not cvmodeoutput:
        recall = acc_num/target_num
        precision = acc_num/predict_num
        F1 = 2*recall*precision/(recall+precision)
        accuracy = acc_num.sum(1)/target_num.sum(1)

        recall = (recall.cpu().numpy()[0]*100).round(3)
        precision = (precision.cpu().numpy()[0]*100).round(3)
        F1 = (F1.cpu().numpy()[0]*100).round(3)
        accuracy = (accuracy.cpu().numpy()[0]*100).round(3)

        print('recall'," ".join('%s' % id for id in recall))
        print('precision'," ".join('%s' % id for id in precision))
        print('F1'," ".join('%s' % id for id in F1))
        print('accuracy',accuracy)

        test_loss /= len(test_loader.dataset)
        # the output is like Test set: Average loss: 0.0163, Accuracy: 6698/10000 (67%)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
             test_loss, correct, len(test_loader.dataset),
             100. * correct / len(test_loader.dataset))) 

        # record the correct predict type
        pred = pred.cpu().numpy()
        pred = pd.DateFrame(pred)
        target.data.cpu().numpy()
        target = pd.DataFrame(target)
        correctpred = pred.loc[pred == target]
        print(correctpred.count())

    return 100. * correct / len(test_loader.dataset)

def relprop(model, relprop_loader):

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
    for i in range(20):
        config(shape=[50,50,50,50],
           classnum=2,
           classnums=False,
           binaryafter=False,
           learningrate=0.0001,
           learningrateschema=optim.SGD,
           batchsize=128,
           epoch=30,
           samplenum=False,
           sampletype=False,
           l1regularization=False,
           l2regularization=False,
           cnn = False,#[[1,1,2,1,0]],
           datapath=f'/data/dataaugmentationinmedicalfield/cv-multi-500k-{i}/',
           batchnorm=0.1,
           dropout=False,
           rawdatatrain=False,
           cvmodeoutput=True,
           cvnum=i,)
