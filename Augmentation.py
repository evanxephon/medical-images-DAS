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

# the strategy here is we first divide the 34 district into 6 lobes, and we use different kernal on each lobe
# and we don't replace data this time, we add them up.
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

# same stratege as upon, but we choose how many to generate 
def generate_different_areas_add_withnum(data,kernelsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1),usedrownum=0):
    dataset = pd.DataFrame(columns=data.columns)
    sampleset = list(range(len(data))
    set1 = random.sample(sampleset,usedrownum)
    for i in set1:
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        augrows = pd.DataFrame(columns=data.columns)
        # it may be same row we are operating, if i've got some extra time, i may fix it
        set2 = random.sample(sampleset,usedrownum)
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

# same strategy as before, but replace not add again
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

# same strategy, but fixed number
def generate_different_areas_replace_withnum(data,kernelsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1),usedrownum=0):
    dataset = pd.DataFrame(columns=data.columns)
    sampleset = list(range(len(data))
    set1 = random.sample(sampleset,usedrownum)
    for i in set1:
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        augrows = pd.DataFrame(columns=data.columns)
        set2 = random.sample(sampleset,usedrownum)
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

# use six fixed kernal to each lobe, but when it come to the type which has very few orginal data, we replace several lobe's data, not
# one, and we calculate the combinations first.
def generate_different_areas_replace_combinations_for_different_type(data,kernelsize=((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)),district=(9,11,4,5,4,1)):
    dataset = pd.DataFrame(columns=data.columns)
                     
    if data.loc[0;'label'] == 0:
        combs = combinations(range(6),1)
    elif data.loc[0;'label'] == 1:
        combs = combinations(range(6),3)
    elif data.loc[0;'label'] == 2:
        combs = combinations(range(6),3)
    elif data.loc[0;'label'] == 3:
        combs = combinations(range(6),3)
    elif data.loc[0;'label'] == 4:
        combs = combinations(range(6),2)
                     
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
                                        thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                                augrows = augrows.append(thechosenrow)
                                thechosenrow = pd.DataFrame(data.iloc[i,:]).T
                    dataset = pd.concat([dataset,augrows])    
    return dataset
                     
# muti-threading
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
            data = self.function(self.data,self.kernelsize,self.num)
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
                        
    # choose the strategy，generate num（optional） kernel size（optional）                         
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
    #config('rawdata1sort.csv',
            function=generate_different_areas_replace,
            num=False,
            testnum=100,
            kernelsize=(((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)),
                        ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
                        ((1,4),(1,1),(1,1),(1,1),(1,1),(1,1)),
                        ((1,5),(1,1),(1,1),(1,1),(1,1),(1,1)),
                        ((1,8),(2,6),(2,1),(2,1),(2,1),(2,1))),
            binary=False,
            savepath='/data/dataaugmentationinmedicalfield/data-2019-3-11-22')
    
    # cross validation
    for x in range(20):
        dirname = 'crossvalidation-'+'batch-1-' + f'{x}'
        config('rawdata1sort.csv',
               function=generate_different_areas_replace,
               num=False,
               testnum=25,
               kernelsize=(((4,9),(4,11),(4,4),(4,5),(4,4),(4,1)),
                           ((1,1),(1,1),(1,1),(1,1),(1,1),(1,1)),
                           ((1,4),(1,1),(1,1),(1,1),(1,1),(1,1)),
                           ((1,5),(1,1),(1,1),(1,1),(1,1),(1,1)),
                           ((1,8),(2,6),(2,1),(2,1),(2,1),(2,1))),
               binary=False,
               savepath='/data/dataaugmentationinmedicalfield/'+dirname)
