# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:14:15 2019

@author: jackl
"""

from math import log
import numpy as np

'''
# static dataset
    dataSet = [['cloudy','good','maybe'],
               ['cloudy','sad','yes'],
               ['cloudy','sad','maybe'],
               ['sunny','good','no'],
               ['sunny','good','no']]
    attriNames = ['Weather','Mood','play or not'] # each column's name.  e.g. attributeNames[i] is i-th column's attribute name
'''
def loadDataSet():
#    dataSet = [['cloudy','good','maybe'],
#           ['cloudy','sad','yes'],
#           ['cloudy','sad','maybe'],
#           ['sunny','good','no'],
#           ['sunny','good','no']]
#    attriNames = ['Weather','Mood','play or not'] # each column's name.  e.g. attributeNames[i] is i-th column's attribute name
#    
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
        
        H(data): a real value which is shannon entropy
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


def infoGain(dataSet,a):
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


def bestAttribute(dataSet):
    '''
    INPUT:
        dataSet: a matrix
    
    OUTPUT:
        best: id of attribute which has biggest information gain
    '''
    r = [infoGain(dataSet,a) for a in range(0,len(dataSet[0])-1)] # 指定不同的属性，返回的信息增益列表
    best = r.index(max(r)) # 返回信息增益最大的相应的列号
    
    return best 

def splitDataset(dataSet,a,v):
    '''
    INPUT:
        dataSet: a matrix
        a : column id
    split dataset by value v in a-th column
    
    OUTPUT:
        dataset...
    '''
    r = []
    for item in dataSet:
        if item[a] == v:
            t = item[:a]            # delete the info of the axis
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
    
    '''
#    print('sfdf')
#    print(dataSet,attriNames)
    if len(dataSet[0])==1: # only 1 column left which means it reaches the last layer
       # print(vote([item[0] for item in dataSet]))
        return vote([item[0] for item in dataSet])
    
    bestCol = bestAttribute(dataSet)
    T = [item[bestCol] for item in dataSet]
    T = set(T)# set based on all bestCol-th column's value
    bestLabel = attriNames[bestCol]
    tree = {bestLabel:{}}
    for item in T:
        r = splitDataset(dataSet,bestCol,item)
      #  print(tree,item)
        
        tree[bestLabel][item] = createTree(r,attriNames[:bestCol]+attriNames[bestCol+1:])
    
    return tree

def decision(new,tree,attriNames):
    print(tree)
    if type(tree) != dict:
        return tree
    # 还要考虑到不是每一个节点下面的分支包括该属性的所有可能值的，因为属性取某个值可能在训练时候一个数据样本都没有
    key = list(tree.keys())[0]
    col = attriNames.index(key)
    value = new[col]
    
    print(new,key,value)
    return decision(new,tree[key][value],attriNames)
    

dataSet,attriNames = loadDataSet()

tree = createTree(dataSet,attriNames)
#
#new = ['sunny','sad']
#
#r = decision(new,tree,attriNames)
new = ['21…50','High','USA']
r = decision(new,tree,attriNames)
print(r)