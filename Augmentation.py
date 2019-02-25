import pandas as pd
import random
from itertools import combinations
import pickle
import threading

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

# 生成所有kernal替换产生的数据 策略为 选择一个固定大小的kernal，
# 每次选择kernal位置的一小块替换成其他数据的相同部分
# kernal也像卷积操作一样移动，我们数据看成 4（年数）* 34（区域数）的矩形
def generate_fixed_kernal(data,kernalsize=(2,28)):
    horizontalsize = 34-kernalsize[1]+1
    verticalsize = 4-kernalsize[0]+1
    dataset = pd.DataFrame(columns=data.columns)
    for i in range(len(data)):
        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        augrows = pd.DataFrame(columns=data.columns)
        for j in range(len(data)):
            if j != i:
                for k in range(horizontalsize):
                    for l in range(verticalsize):
                        a = ((34 * l) + k)
                        b = ((34 * l) + k + kernalsize[1])
                        thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                        thechosenrow.iloc[:, (a+34):(b+34)] = data.iloc[j, (a+34):(b+34)].values
                        augrows = augrows.append(thechosenrow)
                        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        dataset = dataset.append(augrows) 
    return dataset

# 策略为 先将34个区域的数据分成六个大的区域，再在这些小的区域上用kernal
# 但是是将别的数据加上去
def generate_different_areas_add(data,kernalsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1)):
    dataset = pd.DataFrame(columns=data.columns)
    for x in range(len(district)):
        horizontalsize = district[x] - kernalsize[x][1] + 1
        verticalsize = 4 - kernalsize[x][0] + 1
        for i in range(len(data)):
            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
            augrows = pd.DataFrame(columns=data.columns)
            for j in range(len(data)):
                if j != i:
                    for k in range(horizontalsize):
                        for l in range(verticalsize):
                            for m in range(kernalsize[x][1]):
                                a = (sum(district[:x])+m+k)*4 + l
                                b = (sum(district[:x])+m+k)*4 + l + kernalsize[x][0]
                                thechosenrow.iloc[:, a:b] += data.iloc[j, a:b].values
                            augrows = augrows.append(thechosenrow)
                            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
            dataset = pd.concat([dataset,augrows])
    return dataset

# 策略同上，但是指定生成数量
def generate_different_areas_add_withnum(data,kernalsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1)):
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
                horizontalsize = district[x] - kernalsize[x][1] + 1
                verticalsize = 4 - kernalsize[x][0] + 1
                for k in range(horizontalsize):
                    for l in range(verticalsize):
                        for m in range(kernalsize[x][1]):
                            a = (sum(district[:x])+m+k)*4 + l
                            b = (sum(district[:x])+m+k)*4 + l + kernalsize[x][0]
                            thechosenrow.iloc[:, a:b] += data.iloc[j, a:b].values
                        augrows = augrows.append(thechosenrow)
                        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        dataset = pd.concat([dataset,augrows])
    return dataset

# 策略同上上，只是add加变成了replace替换
def generate_different_areas_replace(data,kernalsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1)):
    dataset = pd.DataFrame(columns=data.columns)
    for x in range(len(district)):
        horizontalsize = district[x] - kernalsize[x][1] + 1
        verticalsize = 4 - kernalsize[x][0] + 1
        for i in range(len(data)):
            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
            augrows = pd.DataFrame(columns=data.columns)
            for j in range(len(data)):
                if j != i:
                    for k in range(horizontalsize):
                        for l in range(verticalsize):
                            for m in range(kernalsize[x][1]):
                                a = (sum(district[:x])+m+k)*4 + l
                                b = (sum(district[:x])+m+k)*4 + l + kernalsize[x][0]
                                thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                            augrows = augrows.append(thechosenrow)
                            thechosenrow = pd.DataFrame(data.iloc[i,:]).T
            dataset = pd.concat([dataset,augrows])    
    return dataset

# 策略同上，但指定生成数量
def generate_different_areas_replace_withnum(data,kernalsize=((2,4),(2,5),(2,2),(2,2),(2,2),(2,1)),district=(9,11,4,5,4,1)):
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
                horizontalsize = district[x] - kernalsize[x][1] + 1
                verticalsize = 4 - kernalsize[x][0] + 1
                for k in range(horizontalsize):
                    for l in range(verticalsize):
                        for m in range(kernalsize[x][1]):
                            a = (sum(district[:x])+m+k)*4 + l
                            b = (sum(district[:x])+m+k)*4 + l + kernalsize[x][0]
                            thechosenrow.iloc[:, a:b] = data.iloc[j, a:b].values
                        augrows = augrows.append(thechosenrow)
                        thechosenrow = pd.DataFrame(data.iloc[i,:]).T
        dataset = pd.concat([dataset,augrows])
    return dataset


class outputthread(threading.Thread):
    def __init__(self,function,thetype,data,num=None,kernalsize=None):
        threading.Thread.__init__(self)
        self.function = function
        self.data = data
        self.type = thetype
        self.num = num
        self.kernalsize = kernalsize
    def run(self):
        print('onethreadstart')
        if num and kernalsize:
            data = self.function(self.data,self.num,self.kernalsize)
        elif not num and not kernalsize:
            data = self.function(self.data)
        elif not kernalsize and num:
            data = self.function(self.data,self.num)
        elif kernalsize and not num:
            data = self.function(self.data,self.kernalsize)
        data.to_csv(f'{self.type}.csv',encoding=None,index=False)
        print('onedatafinished')
        
        
if __name__ == '__main__':
    rawdata = pd.read_csv('rawdata0sort.csv')
    data0 = rawdata[rawdata['label'] == 0]
    data1 = rawdata[rawdata['label'] == 1]
    data2 = rawdata[rawdata['label'] == 2]
    data3 = rawdata[rawdata['label'] == 3]
    data4 = rawdata[rawdata['label'] == 4]

    # 训练集和验证集的比例暂时设为4：1，随时可以改动
    # 0    265   200/65
    # 1     49   40/9
    # 2     58   48/10
    # 3     62   48/14
    # 4     73   54/19

    # def partition(data,proportion=0.8):
    '''data0train = data0.iloc[:200, :]
    data1train = data1.iloc[:40, :]
    data2train = data2.iloc[:48, :]
    data3train = data3.iloc[:48, :]
    data4train = data4.iloc[:54, :]'''
    
    data0train = data0.iloc[:245, :]
    data1train = data1.iloc[:29, :]
    data2train = data2.iloc[:38, :]
    data3train = data3.iloc[:42, :]
    data4train = data4.iloc[:52, :]
    
    # 选择生成策略，生成数量（可选）和生成kernal的size（可选）
    thread0 = outputthread(function=,0,data0train,num=,kernalsize=)
    thread1 = outputthread(function=,1,data0train,num=,kernalsize=)
    thread2 = outputthread(function=,2,data0train,num=,kernalsize=)
    thread3 = outputthread(function=,3,data0train,num=,kernalsize=)
    thread4 = outputthread(function=,4,data0train,num=,kernalsize=)
    
    thread0.start()
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    
