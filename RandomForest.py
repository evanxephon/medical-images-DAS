import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

if __name__ == '__main__':
    
    data = pd.read_csv('rawdata.csv')
    
    data0 = data[data['label'] == 0]
    data1 = data[data['label'] == 1]
    data2 = data[data['label'] == 2]
    data3 = data[data['label'] == 3]
    data4 = data[data['label'] == 4]

    #print(len(data0))
    #print(len(data1))
    #print(len(data2))
    #print(len(data3))
    #print(len(data4))

    data0train = data0.iloc[:245,:]
    data0test = data0.iloc[245:,:]
    data1train = data1.iloc[:29,:]
    data1test = data1.iloc[29:,:]
    data2train = data2.iloc[:36,:]
    data2test = data2.iloc[36:,:]
    data3train = data3.iloc[:38,:]
    data3test = data3.iloc[38:,:]
    data4train = data4.iloc[:57,:]
    data4test = data4.iloc[57:,:]

    def getDummy(dataset):
        dataset['sex'] = dataset['sex'].astype('category')
        dataset['scanner'] = dataset['scanner'].astype('category')
        dataset = pd.get_dummies(dataset)
        return dataset
    # traindata = getDummy(data0train.append([data1train,data2train,data3train,data4train]))
    traindata = data0train.append([data1train,data2train,data3train,data4train])
    trainfeature = traindata.drop('label',axis=1)
    traintarget = traindata['label']
    # testdata = getDummy(data0test.append([data1test,data2test,data3test,data4test]))
    testdata = data0test.append([data1test,data2test,data3test,data4test])
    testfeature = testdata.drop('label',axis=1)
    testtarget = testdata['label']
    classifier = RandomForestClassifier(n_estimators=10,max_depth=5,max_features=10)
    classifier.fit(trainfeature,traintarget)
    result = classifier.score(testfeature,testtarget)
    print(result)
