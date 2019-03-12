import pandas as pd
import random
import threading
from sklearn.utils import shuffle
import time
import os

# 生成所有组合的数据，策略为 三个成分组合成一个训练数据
def generate_3X(data, type, component=3):
    output = pd.DataFrame()
    combs = combinations(data.index, component)
    for comb in list(combs):
        tmp1 = pd.DataFrame()
        for index in comb:
            tmp2 = data.loc[index, :].drop('label', axis=0)
            #             以下两行为了区分三种成分，但实际没有必要区分
            #             for i in range(len(tmp2.columns)):
            #                 tmp2.columns[i] = str(tmp2.columns[i]) + f'c{index}'
            tmp1 = pd.concat([tmp1, tmp2], axis=0)
        tmp1 = tmp1.T
        str0 = '-'
        tmp1['label'] = type
        tmp1.index = [str0.join(list(map(str, sorted(comb))))]
        output = output.append(tmp1)
    return output


# 生成指定数量的数据，策略为 三个成分组合成一个训练数据
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

# 生成所有kernel替换产生的数据 策略为 选择一个固定大小的kernel，
# 每次选择kernel位置的一小块替换成其他数据的相同部分
# kernel也像卷积操作一样移动，我们数据看成 4（年数）* 34（区域数）的矩形
def generate_fixed_kernel(data,kernelsize=(2,28)):
    horizontalsize = 34-kernelsize[1]+1
    verticalsize = 4-kernelsize[0]+1
    dataset = pd.DataFrame(columns=data.columns)
    for i in range(len(data)):
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        augrows = pd.DataFrame(columns=data.columns)
        for j in range(len(data)):
            if j != i:
                for k in range(horizontalsize):
                    for l in range(verticalsize):
                        a = ((34 * l) + k)
                        b = ((34 * l) + k + kernelsize[1])
                        thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                        thechosenrow.iloc[:, (a+34):(b+34)] = data.iloc[j, (a+34):(b+34)].values
                        augrows = augrows.append(thechosenrow)
                        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        dataset = dataset.append(augrows) 
    return dataset

# 策略为 先将34个区域的数据分成六个大的区域，再在这些小的区域上用kernel
# 但是是将别的数据加上去
def generate_different_areas_add(data,kernelsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1)):
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
                                thechosenrow.iloc[:, a:b] += data.iloc[j, a:b].values
                            augrows = augrows.append(thechosenrow)
                            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
            dataset = pd.concat([dataset,augrows])
    return dataset

# 策略同上，但是指定生成数量
def generate_different_areas_add_withnum(data,kernelsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1)):
    dataset = pd.DataFrame(columns=data.columns)
    sampleset1 = list(range(200))
    sampleset2 = list(range(200))
    set1 = random.sample(sampleset1,40)
    set2 = random.sample(sampleset2,40)
    for i in set1:
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        augrows = pd.DataFrame(columns=data.columns)
        for j in set2:
            for x in range(len(district)):
                horizontalsize = district[x] - kernelsize[x][1] + 1
                verticalsize = 4 - kernelsize[x][0] + 1
                for k in range(horizontalsize):
                    for l in range(verticalsize):
                        for m in range(kernelsize[x][1]):
                            a = (sum(district[:x])+m+k)*4 + l
                            b = (sum(district[:x])+m+k)*4 + l + kernelsize[x][0]
                            thechosenrow.iloc[:, a:b] += data.iloc[j, a:b].values
                        augrows = augrows.append(thechosenrow)
                        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        dataset = pd.concat([dataset,augrows])
    return dataset

# 策略同上上，只是add加变成了replace替换
def generate_different_areas_replace(data,kernelsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1)):
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
                                thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                            augrows = augrows.append(thechosenrow)
                            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
            dataset = pd.concat([dataset,augrows])    
    return dataset

# 策略同上，但指定生成数量
def generate_different_areas_replace_withnum(data,kernelsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1)):
    dataset = pd.DataFrame(columns=data.columns)
    sampleset1 = list(range(200))
    sampleset2 = list(range(200))
    set1 = random.sample(sampleset1,40)
    set2 = random.sample(sampleset2,40)
    for i in set1:
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        augrows = pd.DataFrame(columns=data.columns)
        for j in set2:
            for x in range(len(district)):
                horizontalsize = district[x] - kernelsize[x][1] + 1
                verticalsize = 4 - kernelsize[x][0] + 1
                for k in range(horizontalsize):
                    for l in range(verticalsize):
                        for m in range(kernelsize[x][1]):
                            a = (sum(district[:x])+m+k)*4 + l
                            b = (sum(district[:x])+m+k)*4 + l + kernelsize[x][0]
                            thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                        augrows = augrows.append(thechosenrow)
                        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        dataset = pd.concat([dataset,augrows])
    return dataset


class outputthread(threading.Thread):
    def __init__(self,function,thetype,data,num=None,classnum='muti',kernelsize=None):
        threading.Thread.__init__(self)
        self.function = function
        self.data = data
        self.type = thetype
        self.num = num
        self.kernelsize = kernelsize
        self.classnum = classnum
    def run(self):
        print('onethreadstart')
        start = time.clock()
        if self.num and self.kernelsize:
            data = self.function(self.data,self.num,self.kernelsize)
        elif not self.num and not self.kernelsize:
            data = self.function(self.data)
        elif not self.kernelsize and self.num:
            data = self.function(self.data,self.num)
        elif self.kernelsize and not self.num:
            data = self.function(self.data,self.kernelsize)
        data.to_csv(f'{self.type}-{self.classnum}.csv',encoding=None,index=False)
        end = time.clock()
        print('onedatafinished')
        print(f'{start-end} seconds for type{self.type} augmentation')
        
def config(data,function,num=False,testnum=100,kernelsize=False,binary=False,savepath=False,crossvalidation=False):
    # choose the data saving path
    if savepath:
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        os.chdir(savepath)
    
    rawdata = pd.read_csv('/data/dataaugmentationinmedicalfield/'+data)
    testdata = pd.DataFrame(columns=rawdata.columns)
    validatedata = pd.DataFrame(columns=rawdata.columns)
    
    data0 = rawdata[rawdata['label'] == 0]
    data1 = rawdata[rawdata['label'] == 1]
    data2 = rawdata[rawdata['label'] == 2]
    data3 = rawdata[rawdata['label'] == 3]
    data4 = rawdata[rawdata['label'] == 4]

    if binary:
        data1 = shuffle(data1.append([data2,data3,data4]))
        data1['label'] = 1
        dataset = [data0,data1]
        testnum = testnum//2
        classnum = 'binary'
    else:
        dataset = [data0,data1,data2,data3,data4]
        testnum = testnum//5
        classnum = 'muti'
        
    # cross validation
    if crossvalidation:
        for x in range(len(dataset)):
            dataset[x] = shuffle(dataset[x])
                        
    # 选择生成策略，生成数量（可选）和生成kernel的size（可选）                         
    for x in range(len(dataset)):  
        datatrain = dataset[x].iloc[:-testnum,:]
        datatest = dataset[x].iloc[-testnum:,:]
        
        testdata = testdata.append(datatest)
        validatedata = validatedata.append(datatrain)
        
    # open a thread
        thread = outputthread(function,x,datatrain,num,classnum=classnum,kernelsize=kernelsize[x])
        thread.start()
        
    # save the testdata
    testdata.to_csv(f'testdata-{classnum}.csv',encoding=None,index=False)
    validatedata.to_csv(f'validatedata-{classnum}.csv',encoding=None,index=False)
    
if __name__ == '__main__':
    #config('rawdata1sort.csv',generate_different_areas_replace,num=False,testnum=100,kernelsize=(((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)),((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),((1,4),(1,1),(1,1),(1,1),(1,1),(1,1)),((1,5),(1,1),(1,1),(1,1),(1,1),(1,1)),((1,8),(2,6),(2,1),(2,1),(2,1),(2,1)))
          ,binary=False,savepath='/data/dataaugmentationinmedicalfield/data-2019-3-11-22')
    #config('rawdata1sort.csv',generate_different_areas_replace,num=False,testnum=100,kernelsize=(((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)),((4,9),(4,11),(4,4),(4,5),(4,4),(4,1))),binary=True)                
    
    # cross validation
    for x in range(100):
        dirname = 'crossvalidation-' + f'{x}'
        config('rawdata1sort.csv',generate_different_areas_replace,num=False,testnum=25,kernelsize=(((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)),((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),((1,4),(1,1),(1,1),(1,1),(1,1),(1,1)),((1,5),(1,1),(1,1),(1,1),(1,1),(1,1)),((1,8),(2,6),(2,1),(2,1),(2,1),(2,1)))
               ,binary=False,savepath='/data/dataaugmentationinmedicalfield/'+dirname)
