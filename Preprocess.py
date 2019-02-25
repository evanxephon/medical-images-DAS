import numpy as np
import pandas as pd
import scipy.io

def preprocess(num):
    data = pd.DataFrame(index=range(num))

    for i in range(34):
        feature = scipy.io.loadmat(f'./rawdata/feature{i+1}.mat')
        rawdata = pd.DataFrame(feature['feature_subjects'])
        rawdata.columns = np.array([0.1,0.2,0.3,0.4])+i+1
        data = pd.concat([data,rawdata],axis=1)

    label = scipy.io.loadmat('./rawdata/group1.mat')
    dflabel = pd.DataFrame(label['group_subjects'])
    dflabel.columns = ['label']
    return pd.concat([data,dflabel],axis=1)

# load data from .mat file
def preprocess1(num):
    data = pd.DataFrame(index=range(num))

    for i in range(34):
        feature = scipy.io.loadmat(f'./diagnosis2/diagnosis_2nd/features_tuo/feature{i+1}.mat')
        rawdata = pd.DataFrame(feature['feature_subjects'])
        rawdata.columns = np.array([0.1, 0.2, 0.3, 0.4]) + i + 1
        rawdata = rawdata.iloc[:num, :]
        data = pd.concat([data, rawdata], axis=1)

    extrafeature = pd.read_csv('./diagnosis2/diagnosis_2nd/demographics.csv')
    data = pd.concat([data, extrafeature], axis=1)

    # put on the label
    label = scipy.io.loadmat('./diagnosis2/diagnosis_2nd/features_tuo/group1.mat')
    dflabel = pd.DataFrame(label['group_subjects'])
    dflabel.columns = ['label']
    return pd.concat([data, dflabel], axis=1)

# manually put the district from the same area together
def preprocess2(num):
    data = pd.DataFrame(index=range(num))

    order = [2,6,7,9,15,16,30,33,34,4,12,14,17,18,19,20,24,27,28,32,5,11,13,21,8,22,25,29,31,3,10,23,26,35]
    order = [x-2 for x in order]

    for i in order:
        feature = scipy.io.loadmat(f'./diagnosis2/diagnosis_2nd/features_tuo/feature{i+1}.mat')
        rawdata = pd.DataFrame(feature['feature_subjects'])
        rawdata.columns = np.array([0.1,0.2,0.3,0.4])+i+1
        # the data offered 's size sometimes changes 
        rawdata = rawdata.iloc[:num,:]

        data = pd.concat([data,rawdata],axis=1)

    extrafeature = pd.read_csv('./diagnosis2/diagnosis_2nd/demographics.csv')
    data = pd.concat([data,extrafeature],axis=1)

    label = scipy.io.loadmat('./diagnosis2/diagnosis_2nd/features_tuo/group1.mat')
    dflabel = pd.DataFrame(label['group_subjects'])
    dflabel.columns = ['label']
    return pd.concat([data,dflabel],axis=1)

preprocess(507).to_csv('rawdata.csv',encoding='utf-8',index=False)
preprocess1(506).to_csv('rawdata0.csv', encoding='utf-8', index=False)
preprocess2(506).to_csv('rawdata0sort.csv',encoding='utf-8',index=False)
