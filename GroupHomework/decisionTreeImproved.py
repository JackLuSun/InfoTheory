# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:08:20 2019

@author: jackl
"""

from math import log

def loadDataSet():  
    dataSet = []
    with open('motorbike.csv','r',encoding='utf8') as f:
        line = f.readline()
        attriNames = line.strip().split(',')
        lines = f.readlines()
        for line in lines:
            dataSet.append(line.strip().split(','))
            
    return dataSet,attriNames

def entropy(data):
    '''
    INPUT: 
        data: a vector not a matrix
    OUTPUT: 
        
        H(data): a real value which is the shannon entropy of data
    '''
    size = len(data)
    count = {}
    for item in data:
        if item in count:
            count[item] += 1
        else:# add a new elelment
            count[item] = 1 
    
    r = 0
    for key in count:# 频率作为概率，按熵的计算公式进行计算
        r += (count[key]/size)*log(size/count[key])/log(2)
    
    return r

def infoGain(dataSet,a):# for ID3
    '''
    INPUT:
        dataSet: a matrix
        a : colum id
    
    calculate information gain based on the a-th attribute
    
    OUTPUT:
        information gain based on the a-th attribute in dataSet
    '''
    count = {}# 用来统计第 a 列中各个不同元素出现的次数
    r = {}#r[i] 表示第 a 列中的第i个不同元素相应的标签集合
    labels = []
    for item in dataSet:# 遍历dataSet 的第 a 列，遍历同时把遇到的标签都记录下来，形成一个列表
        labels.append(item[-1])
        if item[a] in count:
            count[item[a]] += 1
            r[item[a]].append(item[-1])
        else:# add new element and create a list for it
            count[item[a]] = 1
            r[item[a]] = [item[-1]]
    
    size = len(dataSet)
    
    s = 0# 存放条件熵
    for key in count:
        s += count[key]/size*entropy(r[key])      

    return entropy(labels)-s

def infoGainRatio(dataSet,a): # for C4.5
    '''
    INPUT:
        dataSet: a matrix
        a : colum id
    
    calculate information gain ratio based on the a-th attribute
    g_R(D,a) denotes information gain ratio,
    g_R(D,a) = g(D,a)/H_a(D)
    g(D,a) denotes information gain
    H_a(D) denotes entropy of all a-th values
    
    OUTPUT:
        information gain ratio based on the a-th attribute in dataSet
    '''
    g = infoGain(dataSet,a)
    t = [item[a] for item in dataSet]
    H_a = entropy(t)
    
    if H_a !=0:# TODO
        return g/H_a
    return g

def bestAttribute(dataSet):
    '''
    choose the best attribute which has biggest information gain from attributes
    
    INPUT:
        dataSet: a matrix
    
    OUTPUT:
        best: id of attribute which has biggest information gain
    '''
    r = [infoGainRatio(dataSet,a) for a in range(0,len(dataSet[0])-1)] # 指定不同的属性，返回的信息增益列表
    best = r.index(max(r)) # 返回信息增益最大的相应的列号
    
    return best 

def splitDataset(dataSet,a,v):
    '''
    INPUT:
        dataSet: a matrix
        a : column id
        v: a specified value which belong a-th column
    split dataset by value v in a-th column
    
    OUTPUT:
        r: a subdataSet consisted of items whose value of a-th column is v,
        eliminate the a-th column at the same time
    '''
    r = []
    for item in dataSet:
        if item[a] == v:
            t = item[:a]
            t += item[a+1:]
            r.append(t)
    return r

def vote(data):
    '''
    INPUT:
        data: a vector
        return the element which has the biggest frequency of appearance in data
    ''' 
    count = {}
    for item in data:
        if item in count:
            count[item] +=1
        else: 
            count[item] = 0
    # get key which has biggest value
    return max(count,key=count.get)

def createTree(dataSet,attriNames):
    '''
    INPUT: 
        dataSet: a matrix
    OUTPUT:
        a decion tree based on dict structure
    '''
    if len(dataSet[0])==1: # only 1 column left which means it reaches the last layer
        return vote([item[0] for item in dataSet])
    
    bestCol = bestAttribute(dataSet)
    T = [item[bestCol] for item in dataSet]
    T = set(T)# set based on all bestCol-th column's value
    labelOfHighestFreq = vote([item[-1] for item in dataSet])
    bestLabel = (attriNames[bestCol],labelOfHighestFreq)
    tree = {bestLabel:{}}
    for item in T:
        r = splitDataset(dataSet,bestCol,item)
        tree[bestLabel][item] = createTree(r,attriNames[:bestCol]+attriNames[bestCol+1:])
    
    return tree

def decision(new,tree,attriNames):
    '''
        INPUT:
            new: a new sample
            tree: trained decision tree
            attriNames: attributes' name for all columns 
            
        OUTPUT:
            decision result
    '''
    if type(tree) != dict:
        return tree
    # 还要考虑到不是每一个节点下面的分支包括该属性的所有可能值的，因为属性取某个值可能在训练时候一个数据样本都没有
    key = list(tree.keys())[0]# key is a tuple, which consist of two element (attributes,a label which own most frequency among the sub-dataSet)
    name = key[0]# get the attribute's name of the root node
    col = attriNames.index(name)
    value = new[col]

    if value=='': return key[1]# 处理特征缺失的情况
    if value not in tree[key]: return key[1]
    
    return decision(new,tree[key][value],attriNames)

dataSet,attriNames = loadDataSet()

tree = createTree(dataSet,attriNames)


new = ['21…50','High','USA','']#注意，输入必须严格按特征顺序输入，如果样本该特征缺失，则填入空字符串
r = decision(new,tree,attriNames)
print(r)
