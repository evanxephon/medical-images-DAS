import pandas as pd
import torch
import numpy as np
import torch.utils.data
from sklearn.utils import shuffle
import os

# inherit from torch.utils.data.Dataset to realize myown dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.len = len(data)
        self.data = data
        data['label'] = data['label'].astype('int')
        label = data.loc[:,'label']
        feature = data.drop('label',axis=1)
        feature = feature.astype('float')
        self.label = torch.LongTensor(np.array(label))
        self.feature = torch.Tensor(np.array(feature))

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.len


# onehotilize
def getDummy(dataset):
    dataset['sex'] = dataset['sex'].astype('category')
    dataset['scanner'] = dataset['scanner'].astype('category')
    dataset = pd.get_dummies(dataset)
    return dataset

def maketraindata(samplenum=None,sampletype=False,binaryafter=False,classnums=False,traindata=()):
    # sample to get trainset (may not be a necessity)
    traindataset = pd.DataFrame()

    for x in range(len(traindata)):
        traindatax = pd.read_csv(traindata[x])

        for feature in ['sex','visit_age','scanner']:
            if feature in traindatax.columns:
                traindatax.drop(columns=feature,inplace=True)

        if sampletype == 'up':
            traindatax = upsample(traindatax,samplenum)
        elif sampletype == 'down':
            traindatax = downsample(traindatax,samplenum) 

        print(f'type{x} has {len(traindatax)} data')
        traindataset = traindataset.append(traindatax) 

    if classnums:
        for i in range(len(classnums)):
            traindataset.loc[traindataset['label'] == classnums[i], 'label'] = i

    if binaryafter:
        traindataset.loc[traindataset['label'] > 1 ,'label'] = 1 

    return  traindataset

# fill the NaN with mean value or something else
def fillnan(data):

    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = feature[column].mean()
        data[column].fillna(mean_val, inplace=True)

    return data

def upsample(data,num):
    return data.sample(num,replace=True)

def downsample(data,num):
    return data.sample(num)

def config(batchsize,traindata,validatedata,testdata,classnum,classnums,binaryafter=False,onehot=True):

    # set the batchsize and the other things
    batch_size = batchsize
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # onehot
    if onehot:
        traindata = getDummy(traindata)
        validatedata = getDummy(validatedata)
        testdata = getDummy(testdata)

    # get balanced validatedata
    validatedatabalanced = pd.DataFrame()
    
    
    for x in range(classnum):
        validatedatabalanced = validatedatabalanced.append(validatedata[validatedata['label'] == x].sample(500 // classnum ,replace=True))
    
    for feature in ['sex','visit_age','scanner']:
        if feature in validatedatabalanced.columns:
            validatedatabalanced.drop(columns=feature,inplace=True)

        if feature in testdata.columns:
            testdata.drop(columns=feature,inplace=True)

    if binaryafter:
        validatedata.loc[validatedata['label'] > 1, 'label'] = 1
        testdata.loc[testdata['label'] > 1, 'label'] = 1

    if classnums:
        for i in range(5):
            if i not in classnums:
                validatedata = validatedata.loc[validatedata['label'] != i]
                testdata = testdata.loc[testdata['label'] != i]

        for i in range(len(classnum)):
            validatedata.loc[validatedata['label'] == classnums[i], 'label'] = i
            testdata.loc[testdata['label'] == classnums[i], 'label'] = i
  

    traindata = MyDataset(traindata)
    validatedata = MyDataset(validatedatabalanced)
    testdata = MyDataset(testdata)

    # set the dataloader api
    train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                               batch_size=batch_size,
                                               shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=validatedata,
                                               batch_size=batch_size,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=testdata,
                                              batch_size=batch_size,
                                              shuffle=False)
    relprop_loader = torch.utils.data.DataLoader(dataset=testdata,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader,validate_loader,test_loader,relprop_loader

def getloader(samplenum,sampletype,batchsize,traindata,validatedata,testdata,classnum,classnums,binaryafter=False,datapath=False,rawdatatrain=False):

    if datapath:
        os.chdir(datapath) 
    if binaryafter:
        testdata = 'testdata-multi.csv'
        validatedata = 'validatedata-multi.csv'
    testdata = pd.read_csv(testdata)
    validatedata = pd.read_csv(validatedata)

    if rawdatatrain:
        traindata = pd.DataFrame()
        if samplenum:
	    for x in range(classnum):
                traindata = traindata.append(validatedata[validatedata['label'] == x])
    else:
        traindata = maketraindata(samplenum,sampletype,binaryafter=binaryafter,classnums=classnums,traindata=traindata)

    return config(batchsize,traindata,validatedata,testdata,classnum,classnums,binaryafter=binaryafter,onehot=False)
