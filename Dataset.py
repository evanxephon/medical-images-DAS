import pandas as pd
import torch
import numpy as np
import torch.utils.data

# 继承torch.utils.data.Dataset实现自己的数据集类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.len = len(data)
        self.data = data
        label = data.loc[:,'label']
        feature = data.drop('label',axis=1)
        for column in list(feature.columns[feature.isnull().sum() > 0]):
            mean_val = feature[column].mean()
            feature[column].fillna(mean_val, inplace=True)
        self.label = torch.LongTensor(np.array(label))
        self.feature = torch.Tensor(np.array(feature))
    def __getitem__(self, index):
        return self.feature[index],self.label[index]

    def __len__(self):
        return self.len

# onehot
def getDummy(dataset):
    dataset['sex'] = dataset['sex'].astype('category')
    dataset['scanner'] = dataset['scanner'].astype('category')
    dataset = pd.get_dummies(dataset)
    return dataset

def maketraindata(num,*traindata):
    # 训练集
    data0 = pd.read_csv(traindata[0]).sample(num)
    data1 = pd.read_csv(traindata[1]).sample(num)
    data2 = pd.read_csv(traindata[2]).sample(num)
    data3 = pd.read_csv(traindata[3]).sample(num)
    data4 = pd.read_csv(traindata[4]).sample(num)

    traindata = data0.append([data1, data2, data3, data4])
    return  traindata

def maketestdata(num,rawdata):

    # 设置训练集和验证集
    data0 = rawdata[rawdata['label'] == 0]
    data1 = rawdata[rawdata['label'] == 1]
    data2 = rawdata[rawdata['label'] == 2]
    data3 = rawdata[rawdata['label'] == 3]
    data4 = rawdata[rawdata['label'] == 4]

    data0test = data0.iloc[-num:, :]
    data1test = data1.iloc[-num:, :]
    data2test = data2.iloc[-num:, :]
    data3test = data3.iloc[-num:, :]
    data4test = data4.iloc[-num:, :]

    # 测试数据做过采样操作来达到平衡数据集状态
    # testdata0 = data0test.loc[np.random.choice(data0test.index,size=num, replace=True),:]
    # testdata1 = data1test.loc[np.random.choice(data1test.index,size=num, replace=True),:]
    # testdata2 = data2test.loc[np.random.choice(data2test.index,size=num, replace=True),:]
    # testdata3 = data3test.loc[np.random.choice(data3test.index,size=num, replace=True),:]
    # testdata4 = data4test.loc[np.random.choice(data4test.index,size=num, replace=True),:]
    testdata = data0test.append([data1test,data2test,data3test,data4test])

    return testdata

def config(traindata,testdata):
    # 设置每一批训练数据的大小
    batch_size = 128
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train_dataset = MyDataset(getDummy(traindata))
    test_dataset = MyDataset(getDummy(testdata))

    # 构造自定义数据读取接口
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader,test_loader

def getloader(trainnum=100000,testnum=20,rawdata=pd.read_csv('rawdata0sort.csv'),*traindata):
    traindata = maketraindata(trainnum,traindata)
    testdata = maketestdata(testnum,rawdata)
    return config(traindata,testdata)
