# -*- coding: utf-8 -*-
from numpy import *
#样本数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0,1,0,1,0,1]    #类别标签 1侮辱性文字 0正常文字
    return postingList,classVec    #返回文档向量类别向量


#创建(不重复)词汇表
# 输入：dataSet已经经过切分处理
# 输出：包含所有文档中出现的不重复词的列表

def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空集合， 构建set集合，会返回不重复词表
    for document in dataSet:        # #遍历每篇文档向量，扫描所有文档的单词
        # 通过set(document)，获取document中不重复词列表
       vocabSet = vocabSet | set(document) #求并集
    return list(vocabSet)

#***词集模型：只考虑单词是否出现
#vocabList：词汇表
#inputSet ：某个文档向量

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)            #创建所含元素全为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 #依次取出文档中的单词与词汇表进行对照，若在词汇表中出现则为1
        else: print "the word: %s is not in my Vocabulary!" % word #若测试文档的单词，不在词汇表中，显示提示信息，该单词出现次数用0表示
    return returnVec

#=====训练分类器，优化处理=====
#输入trainMatrix：词向量数据集
#输入trainCategory：数据集对应的类别标签
#输出p0Vect：词汇表中各个单词在正常言论中的类条件概率密度
#输出p1Vect：词汇表中各个单词在侮辱性言论中的类条件概率密度
#输出pAbusive：侮辱性言论在整个数据集中的比例


def trainNB0(trainMatrix,trainCategory):  #trainCategory为每篇文档类别所构成的向量
    numTrainDocs = len(trainMatrix)            #训练集总条数：行数
    numWords = len(trainMatrix[0])                #   #训练集中所有单词总数：词向量维度
    pAbusive = sum(trainCategory)/float(numTrainDocs)       #侮辱类的概率(侮辱类占总训练数据的比例)
    p0Num = ones(numWords); p1Num = ones(numWords)         #*正常言论的类条件概率密度 p(某单词|正常言论)=p0Num/p0Denom
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):          ##遍历训练集每个样本
        if trainCategory[i] == 1:             # #若为侮辱类
            p1Num += trainMatrix[i]                 #统计侮辱类所有文档中的各个单词总数
            p1Denom += sum(trainMatrix[i])          # p1Denom侮辱类总单词数
            # 若为正常类
        else:
            p0Num += trainMatrix[i]         ##统计正常类所有文档中的各个单词总数
            p0Denom += sum(trainMatrix[i])      ## #p0Denom正常类总单词数
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)             #数据取log，即单个单词的p(x1|c1)取log，防止下溢出
    return p0Vect,p1Vect,pAbusive

#vec2Classify：待分类文档
#p0Vect:词汇表中每个单词在训练样本的正常言论中的类条件概率密度
#p1Vect:词汇表中每个单词在训练样本的侮辱性言论中的类条件概率密度
#pClass1：侮辱性言论在训练集中所占的比例
#在对数空间中进行计算，属于哪一类的概率比较大就判为哪一类

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0


# ***词袋模型：考虑单词出现的次数
#  vocabList：词汇表
# inputSet ：某个文档向量
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)     ##创建所含元素全为0的向量
    # 依次取出文档中的单词与词汇表进行对照，统计单词在文档中出现的次数
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1    ##单词在文档中出现的次数
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet()    # #获得训练数据，类别标签
    myVocabList = createVocabList(listOPosts)   # #创建词汇表
    trainMat=[]     # #构建矩阵，存放训练数据
  # 遍历原始数据，转换为词向量，构成数据训练矩阵
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  ##  #数据转换后存入数据训练矩阵trainMat中
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))   ## #训练分类器

    # ===测试数据（1）
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))   # #测试数据转为词向量
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)   ###输出分类结果

    # ===测试数据（2）
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))##测试数据转为词向量
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)  #输出分类结果


