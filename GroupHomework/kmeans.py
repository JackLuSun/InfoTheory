# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:59:04 2019

@author: jackl
"""
from math import log

def divergence(a,b):
    '''
        INPUT: 
            a: a vector
            b: a vector having same dimension with b
        OUTPUT:
            KL Divergence between a and b
    '''
    r = 0
    for i in range(0,len(a)):
        r += a[i]*log((a[i]+1)/(b[i]+1))/log(2)
    
    return r

def dist(a,b):
    '''
        INPUT:
            a: a vector which represents a document, a[i] is the importance of i-th word in the document
            b: just like a
        return averaged K-L divergence between a and b
    '''
    lam = 0.5# see averaged K-L divergence
    M = [lam*a[i]+(1-lam)*b[i] for i in range(0,len(a))]
    
    r = lam*divergence(a,M) + (1-lam)*divergence(b,M)    
    
    return r

def kmean(dataSet, k = 5):
    if k > len(dataSet): return None
    # 选择前面 k 个样本作为初始 k个中心点
    u = dataSet[0:k]
    
    r = [0]*len(dataSet) # r[i]中存放的是类别号： 表示第 i 个样本被归类为 第 r[i] 个类别
    
    # 对每个样本进行归类
    while True:
        for j in range(0,len(dataSet)):
            # 计算样本 item 到每个中心的距离，并将其归属于最小距离的那个中心相应的类别
            t = 0
            for i in range(0,len(u)):
                if dist(dataSet[j],u[i]) < dist(dataSet[j],u[t]):
                    t = i
            r[j] = t
        # 所有样本归属完毕后，根据归属情况，重新计算k 个簇中心
        count = [0]*k # count[i] 表示 第 i 个类别中有多少个元素
        uu = [[0]*len(dataSet[0]) for i in range(0,k)]
        for i in range(0,len(dataSet)):
            count[r[i]] += 1
            uu[r[i]] = [uu[r[i]][j] + dataSet[i][j] for j in range(0,len(dataSet[0]))]
        print(count)
        for i in range(0,k):
            uu[i] = [uu[i][j]/count[i] for j in range(len(uu[i]))]
        if uu == u: return r # 新旧中心没有发生变化，故而训练完毕
        u = uu# 更新中心，继续训练
        
    return None

dataSet = [[1,2,3,4],[1,3,3,4],[33,1,2,2],[33,4,7,7],[43,32,12,33],[55,30,11,33]]

r = kmean(dataSet,k=3)
print(r)
        
    
            