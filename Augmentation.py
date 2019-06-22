import pandas as pd
import random
import threading
from itertools import combinations
from sklearn.utils import shuffle
import time
import os

# combine some row of data into one single row
def generate_3X(data, type, component=3):
    output = pd.DataFrame()
    combs = combinations(data.index, component)
    for comb in combs:
        tmp1 = pd.DataFrame()
        for index in comb:
            tmp2 = data.loc[index, :].drop('label', axis=0)
            #             these commented code is used for distinguish the component, but it seems not necessary
            #             for i in range(len(tmp2.columns)):
            #                 tmp2.columns[i] = str(tmp2.columns[i]) + f'c{index}'
            tmp1 = pd.concat([tmp1, tmp2], axis=0)
        tmp1 = tmp1.T
        str0 = '-'
        tmp1['label'] = type
        tmp1.index = [str0.join(list(map(str, sorted(comb))))]
        output = output.append(tmp1)
    return output


# it's also combine, but we set the number of generating data
def generate_3X_withnum(data, size, type, component=3):
    output = pd.DataFrame()
    for i in range(size):
        tmp1 = pd.DataFrame()
        combination = random.sample(range(data.index[0], data.index[0] + len(data)), component)
        for j in range(len(combination)):
            tmp2 = data.loc[combination[j], :].drop('label', axis=0)
            # tmp2.columns = tmp2.columns + f'c{j}'
            tmp1 = pd.concat([tmp1, tmp2], axis=0)
        tmp1 = tmp1.T
        str0 = '-'
        tmp1['label'] = type
        tmp1.index = [str0.join(list(map(str, sorted(combination))))]
        output = output.append(tmp1)
    return output

# we use a kernal which we only focus on it's shape,
# it is like a convolution process to some extent, but we just replace the data in the kernal by the data of another row in the same position, 
# in these process, the data is seen as a matrix 4 years * 34 districts
def generate_fixed_kernel(data,kernelsize=(2,28), strategy='replace'):
    horizontalsize = 34-kernelsize[1]+1
    verticalsize = 4-kernelsize[0]+1
    dataset = pd.DataFrame(columns=data.columns)
    for i in range(len(data)):
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        for j in range(len(data)):
            augrows = pd.DataFrame(columns=data.columns)
            if j != i:
                for k in range(horizontalsize):
                    for l in range(verticalsize):
                        a = ((34 * l) + k)
                        b = ((34 * l) + k + kernelsize[1])
                        
                        if strategy == 'replace':
                            thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                            thechosenrow.iloc[:, (a+34):(b+34)] = data.iloc[j, (a+34):(b+34)].values
                        elif strategy == 'add':
                            thechosenrow.iloc[:, a:b] += data.iloc[j, a:b].values
                            thechosenrow.iloc[:, (a+34):(b+34)] += data.iloc[j, (a+34):(b+34)].values
                            
                        augrows = augrows.append(thechosenrow)
                        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
            dataset = dataset.append(augrows) 
    return dataset


# the strategy here is we first divide the 34 district into 6 lobes, and we use different kernal on each lobe
# and we can either replace data, or add them up.
def generate_different_kernels(data,kernelsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1),strategy='replace'):
    dataset = pd.DataFrame(columns=data.columns)
    for x in range(len(district)):
        horizontalsize = district[x] - kernelsize[x][1] + 1
        verticalsize = 4 - kernelsize[x][0] + 1
        for i in range(len(data)):
            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
            augrows = pd.DataFrame(columns=data.columns)
            for j in range(len(data)):
                if j != i:
                    for k in range(horizontalsize):
                        for l in range(verticalsize):
                            for m in range(kernelsize[x][1]):
                                a = (sum(district[:x])+m+k)*4 + l
                                b = (sum(district[:x])+m+k)*4 + l + kernelsize[x][0]
                                
                                if strategy == 'replace':
                                    thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                                elif strategy == 'add':
                                    thechosenrow.iloc[:, a:b] += data.iloc[j, a:b].values
                         
                            augrows = augrows.append(thechosenrow)
                            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
                            
            dataset = dataset.append(augrows)  
    return dataset

# same strategy, but fixed number
def generate_different_kernels_withnum(data,kernelsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1),usedrownum=0,strategy='replace'):
    dataset = pd.DataFrame(columns=data.columns)
    sampleset = list(range(len(data)))

    for i in range(len(data)):
        
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        augrows = pd.DataFrame(columns=data.columns)
        chosenset = random.sample(sampleset,usedrownum)
        
        for j in chosenset:
            for x in range(len(district)):
                horizontalsize = district[x] - kernelsize[x][1] + 1
                verticalsize = 4 - kernelsize[x][0] + 1
                for k in range(horizontalsize):
                    for l in range(verticalsize):
                        for m in range(kernelsize[x][1]):
                            a = (sum(district[:x])+m+k)*4 + l
                            b = (sum(district[:x])+m+k)*4 + l + kernelsize[x][0]
                            
                            if strategy == 'replace':
                                thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                            elif strategy == 'add':
                                thechosenrow.iloc[:, a:b] += data.iloc[j, a:b].values
                                
                        augrows = augrows.append(thechosenrow)
                        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
                        
        dataset = dataset.append(augrows)
    return dataset

# use six fixed kernal to each lobe, but when it come to the type which has very few orginal data, we replace or add several lobe's data, not
# one, and we calculate the combinations first.
def generate_different_kernels_combinations_for_different_type(data,kernelsize=((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)),district=(9,11,4,5,4,1),strategy='replace'):
    dataset = pd.DataFrame(columns=data.columns)
                     
    if data.iloc[0,-1] == 0:
        combs = list(combinations(range(6),1))
    elif data.iloc[0,-1] == 1:
        combs = list(combinations(range(6),3))
    elif data.iloc[0,-1] == 2:
        combs = list(combinations(range(6),3))
    elif data.iloc[0,-1] == 3:
        combs = list(combinations(range(6),3))
    elif data.iloc[0,-1] == 4:
        combs = list(combinations(range(6),2))
                     
    for i in range(len(data)):
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        augrows = pd.DataFrame(columns=data.columns)
        for j in range(len(data)):
            if j != i:
                for comb in combs:
                    
                    horizontalsize = district[comb[0]] - kernelsize[comb[0]][1] + 1
                    verticalsize = 4 - kernelsize[comb[0]][0] + 1
                    
                    for k in range(horizontalsize):
                        for l in range(verticalsize):
                            for x in comb:
                                for m in range(kernelsize[x][1]):
                                    a = (sum(district[:x])+m+k)*4 + l
                                    b = (sum(district[:x])+m+k)*4 + l + kernelsize[x][0]
                                    
                                    if strategy == 'replace':
                                        thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                                    elif strategy == 'add':
                                        thechosenrow.iloc[:, a:b] += data.iloc[j, a:b].values
                                        
                            augrows = augrows.append(thechosenrow)
                            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
                            
        dataset = dataset.append(augrows)    
    return dataset
                     
# multi-threading
class outputthread(threading.Thread):
    def __init__(self,function,thetype,data,num=None,classnum='multi',kernelsize=None,strategy='replace'):
        threading.Thread.__init__(self)
        self.function = function
        self.data = data
        self.type = thetype
        self.num = num
        self.kernelsize = kernelsize
        self.classnum = classnum
        self.strategy = strategy
    def run(self):
        print('onethreadstart')
        start = time.clock()
        if self.num and self.kernelsize:
            data = self.function(self.data,self.kernelsize,self.num,strategy=self.strategy)
        elif not self.num and not self.kernelsize:
            data = self.function(self.data,strategy=self.strategy)
        elif not self.kernelsize and self.num:
            data = self.function(self.data,self.num,strategy=self.strategy)
        elif self.kernelsize and not self.num:
            data = self.function(self.data,self.kernelsize,strategy=self.strategy)
        data.to_csv(f'{self.type}-{self.classnum}.csv',encoding=None,index=False)
        end = time.clock()
        print('onedatafinished')
        print(f'{start-end} seconds for type{self.type} augmentation')
        
def config(data,function,num=False,testnum=100,kernelsize=False,binary=False,savepath=False,cv_order=1,cv_shuffle=1,cv_fold=1,thread=False,strategy='replace'):
    
    rawdata = pd.read_csv('/data/dataaugmentationinmedicalfield/'+data)
    
    data0 = rawdata[rawdata['label'] == 0]
    data1 = rawdata[rawdata['label'] == 1]
    data2 = rawdata[rawdata['label'] == 2]
    data3 = rawdata[rawdata['label'] == 3]
    data4 = rawdata[rawdata['label'] == 4]

    if binary:
        data1 = data1.append([data2,data3,data4])
        data1['label'] = 1
        dataset = [data0,data1]
        testnum = testnum//2
        classnum = 'binary'
    else:
        dataset = [data0,data1,data2,data3,data4]
        testnum = testnum//5
        classnum = 'multi'
        
    # cross validation, three ways to do: 1.shuffle, 2.order, 3.fold
    
    trainset = []
    testset = []
    
    for x in range(len(dataset)):
        start = 0

        if cv_shuffle != 1:
            end = testnum

        elif cv_fold != 1:
            interval = 505 // cv_fold // 2
            end = interval

        elif cv_order != 1:
            end = testnum
            
        traindata = []
        testdata = []
        
        for i in range(cv_shuffle):
            for j in range(cv_fold):
                for k in range(cv_order):

                    if cv_shuffle != 1:
                        dataset[x] = shuffle(dataset[x])
                            
                    if start < end:
                        datatrain = pd.concat([dataset[x].iloc[0:start,:], dataset[x].iloc[end:,:]], axis=0)
                        datatest = dataset[x].iloc[start:end,:]
                    else:
                        datatrain = dataset[x].iloc[end:start,:]
                        datatest = pd.concat([dataset[x].iloc[0:end,:], dataset[x].iloc[start:,:]], axis=0)

                    testdata.append(datatest)
                    traindata.append(datatrain)

                    if cv_fold != 1:
                        end += interval
                        start += interval

                    if cv_order != 1:
                        start = end
                        end = (start + testnum)%len(dataset[x])
                        
        trainset.append(traindata)
        testset.append(testdata)
    
    # create the zip generator for the iteration
    if binary:
        trainzip = zip(trainset[0], trainset[1])
        testzip = zip(testset[0], testset[1])
    else:
        trainzip = zip(trainset[0], trainset[1], trainset[2], trainset[3], trainset[4])
        testzip = zip(testset[0], testset[1], testset[2], testset[3], testset[4])
    
    batch = 0
    for traindata, testdata in zip(trainzip, testzip):
        
        savedir = savepath + f'{batch}'
        batch += 1
        
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        os.chdir(savedir)

        for x in range(len(traindata)):
        #open a thread
            if thread:
                if kernelsize:
                    thread = outputthread(function,x,traindata[x],num[x],classnum=classnum,kernelsize=kernelsize[x],strategy=strategy)
                else:
                    thread = outputthread(function,x,traindata[x],num[x],classnum=classnum,strategy=strategy)
                thread.start()
            else:
                if num and kernelsize:
                    data = function(traindata[x],kernelsize[x],usedrownum=num[x],strategy=strategy)
                elif not num and not kernelsize:
                    data = function(traindata[x],strategy=strategy)
                elif not kernelsize and num:
                    data = function(traindata[x],usedrownum=num[x],strategy=strategy)
                elif kernelsize and not num:
                    data = function(traindata[x],kernelsize[x],strategy=strategy)
                data.to_csv(f'{x}-{classnum}.csv',encoding=None,index=False)
                
        traindata = pd.concat(traindata, axis=0)
        testdata = pd.concat(testdata, axis=0)
        testdata.to_csv(f'testdata-{classnum}.csv',encoding=None,index=False)
        traindata.to_csv(f'validatedata-{classnum}.csv',encoding=None,index=False)
    
if __name__ == '__main__':
    config('rawdata2sort.csv',
            function=generate_different_kernels_withnum,
            #num=[16, 18],# for 50k binary data
            #num=[81, 90],# for 250k binary data
            #num=[163, 181],# for 500k bin data
            num=[10, 43, 50, 52, 54],
            #num=[2, 12, 10, 10, 8],
            testnum=25,
            #kernelsize=(((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)),
            #            ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
            #            ((1,4),(1,1),(1,1),(1,1),(1,1),(1,1)),
            #            ((1,5),(1,1),(1,1),(1,1),(1,1),(1,1)),
            #            ((1,8),(2,6),(2,1),(2,1),(2,1),(2,1))),
            kernelsize = list(((2,9),(2,11),(2,4),(2,5),(2,4),(2,1)) for x in range(5)),
            #kernelsize = list(((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)) for x in range(2)),
            binary=True,
            savepath='/data/dataaugmentationinmedicalfield/cv-multi-500k-p-',
            cv_order=20,
            cv_shuffle=1,
            cv_fold=1,
            thread=False,
            strategy='replace')

