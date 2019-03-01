import pandas as pd
import torch
import numpy as np
import torch.utils.data
from sklearn.utils import shuffle


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
        return self.feature[index],self.label[index]

    def __len__(self):
        return self.len


# onehotilize
def getDummy(dataset):
    dataset['sex'] = dataset['sex'].astype('category')
    dataset['scanner'] = dataset['scanner'].astype('category')
    dataset = pd.get_dummies(dataset)
    return dataset

def maketraindata(num,traindata):
    # sample to get trainset (may not be a necessity)
    print(traindata)
    traindataset = pd.DataFrame()
    for data in traindata:
        traindatax = pd.read_csv(data)#.sample(num)
        traindataset = traindataset.append(traindatax)
    print(traindataset.describe()) 
    return  traindataset

def fillnan(data):
    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = feature[column].mean()
        data[column].fillna(mean_val, inplace=True)
    return data

def upsample(data,num):
    return data = data.loc[np.random.choice(data.index,size=num, replace=True),:]

def config(traindata,testdata,onehot=True):
    # set the batchsize and the other things
    batch_size = 64
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if onehot:
        traindata = MyDataset(getDummy(traindata))
        testdata = MyDataset(getDummy(testdata))
    else:
        traindata = MyDataset(traindata)
        testdata = MyDataset(testdata)
    # set the dataloader api
    train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testdata,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader,test_loader

def getloader(trainnum,testdata,traindata):
    traindata = maketraindata(trainnum,traindata)
    testdata = pd.read_csv(testdata)
    return config(traindata,testdata,onehot=False)
