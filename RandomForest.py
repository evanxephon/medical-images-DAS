import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def rf_multi_dataset(rawdata, testnum, upsample=False):
    # multi classification
    traindata = []
    testdata = []

    for raw in rawdata:
        trainset = []
        testset = []
        testnum = testnum
        start = 0
        end = 0
        for time in range(20):
            end = (start + testnum) % len(raw)

            if start < end:
                train = pd.concat([raw.iloc[0:start,:], raw.iloc[end:,:]], axis=0)
                test = raw.iloc[start:end,:]
            else:
                train = raw.iloc[end:start,:]
                test = pd.concat([raw.iloc[0:end,:], raw.iloc[start:,:]], axis=0)
            if upsample:
                train = train.sample(upsample, replace=True)
            trainset.append(train)

            testset.append(test)

        traindata.append(trainset)
        testdata.append(testset)
        
        return traindata, testdata
 
def rf_bin_dataset(rawdatab, testnum, upsample):
    
    # binary classification
    traindatab = []
    testdatab = []
    testnum = testnum
    start = 0
    end = 0
    
    for raw in rawdatab:
        trainset = []
        testset = []
        for time in range(20):
            end = (start + testnum) % len(raw)

            if start < end:
                train = pd.concat([raw.iloc[0:start,:], raw.iloc[end:,:]], axis=0)
                test = raw.iloc[start:end,:]
            else:
                train = raw.iloc[end:start,:]
                test = pd.concat([raw.iloc[0:end,:], raw.iloc[start:,:]], axis=0)
            if upsample:
                train = train.sample(upsample, replace=True)
            trainset.append(train)
            testset.append(test)

            start = end 

        traindatab.append(trainset)
        testdatab.append(testset)
         
       return traindatab, testdatab

def getDummy(dataset):
    for x in dataset.columns:
        if isinstance(x, str):
            dataset[x] = dataset[x].astype('category')
    dataset = pd.get_dummies(dataset)
    return dataset

def randomforest(trainset, testset, binary=False, get_dummy=False,config=False):
    
    results = []
    if binary:
        trainzip = zip(trainset[0], trainset[1])
        testzip = zip(testset[0], testset[1])
    else:
        trainzip = zip(trainset[0], trainset[1], trainset[2], trainset[3], trainset[4])
        testzip = zip(testset[0], testset[1], testset[2], testset[3], testset[4])
    
    for train, test in zip(trainzip, testzip):

        traindata = pd.concat(train, axis=0)
        testdata = pd.concat(test, axis=0)
        
        if get_dummy:
            traindata = getDummy(traindata)
            testdata = getDummy(testdata)
            
        trainx = traindata.drop('label', axis=1)
        trainy = traindata['label']
        testx = testdata.drop('label', axis=1)
        testy = testdata['label']

        rf = RandomForestClassifier(n_estimators=100,max_depth=5,max_features=10)
        rf.fit(trainx, trainy)

        result = rf.score(testx, testy)
        finalresult.append(result)
    
if __name__ == '__main__':
    
    data = pd.read_csv('rawdata2sort.csv')
    
    data0 = data[data['label'] == 0]
    data1 = data[data['label'] == 1]
    data2 = data[data['label'] == 2]
    data3 = data[data['label'] == 3]
    data4 = data[data['label'] == 4]
    
    #multi classification
    dataset = [data0, data1, data2, data3, data4]
    traindata, testdata = rf_multi_dataset(dataset, 5, 500)
    result_multi = randomforest(traindata, testdata)
    
    #binary classification
    data1 = pd.concat([data1,data2,data3,data4])
    data1.loc[:,'label'] = 1
    dataset = [data0, data1]
    traindata, testdata = rf_bin_dataset(dataset, 10, 500)
    result_bin = randomforest(traindata, testdata)

