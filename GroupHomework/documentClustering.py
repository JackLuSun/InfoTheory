# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:47:36 2019

@author: jackl
"""

import csv
import re
from math import log

COMMENT = True

def removePunctuation(text):
    '''
        INPUT:
            text: a string
        remove punctuation in text
        
        OUTPUT: a string   
    '''
    punctuation = '!,;:?"\'.'
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

def loadDataSet():
    '''
        load data from given .csv file
        OUTPUT:
            dataSet: 
                i-th paper is represented as dataSet[i] which consists of 
                all words from title, keyword and abstract in i-th paper
                
            wordVector: a list which stores the SET of all words from all paper 's title,keyword and abstract
    '''
    dataSet = []
    a = 'AAAI-14 Accepted Papers.csv'
    #b = 'test.csv'
    #c= 'test2.csv'
    with open(a) as f:
        items = csv.reader(f)
        next(items)# skip header
        wordVector = []
        for item in items:
            #print(item[5])
            t = item[0]+" "+item[3]+" "+item[5]# title,keyword, abstract
            t = removePunctuation(t).split()
            dataSet.append(t)
            wordVector += t
       
    return dataSet,list(set(wordVector))

def dfCount(dataSet,wv):
    '''
        document frequency based on a specified word
        df[i] means the frequence of documents which have the i-th word in wordVector wv
        
        OUTPUT:
            df: a vector
    '''
    df = [0]*len(wv)
    for i in range(0,len(wv)):
        for doc in dataSet:
          #  print(wv)
            if wv[i] in doc:
                df[i] += 1
    df = [v/len(dataSet) for v in df]
    return df

def tfCount(dataSet,wv):
    '''
        tf[i] means frequence of word in wv in the i-th doc in dataSet
        tf[i][j] means frequence of j-th word in wv in the i-th doc in dataSet
    '''
    tf = []
    for doc in dataSet:
        tf.append([doc.count(word)/len(doc) for word in wv])
    
    return tf

def tf_idfCalc(tf,df):
    '''
    OUTPUT:
        tf_idf[i][j] means importance of j-th word in i-th document
    '''
    tf_idf = []
    for i in range(0,len(tf)):
        tf_idf.append([tf[i][j]*log(1/df[j]) for j in range(0,len(df))])
    
    return tf_idf

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
    '''
        INPUT:
            dataSet: dataSet[i] represents the i-th document which is a vector
            k: k clusters
        OUTPUT:
            r: r[i] means the i-th document in r[i]-th cluster
    '''
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
        #print(count)
        for i in range(0,k):
            uu[i] = [uu[i][j]/count[i] for j in range(len(uu[i]))]
        if uu == u: return r # 新旧中心没有发生变化，故而训练完毕
        u = uu# 更新中心，继续训练
        
    return None
    
    

dataSet,wordVector = loadDataSet()
tf = tfCount(dataSet,wordVector)
df = dfCount(dataSet,wordVector)

tf_idf = tf_idfCalc(tf,df)

r = kmean(tf_idf,5)
print(r)
