'''
该函数是数据集的预设
'''
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
'''
从数据集中构建C1是大小为1的候选集合，
'''
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    C1 = [frozenset(i) for i in C1]    #为C1中的每个类别创建不变集合
    return  C1

'''
扫描C1判断只有一个元素的项集是否满足最小支持度的要求
对满足最小支持度的项集构成集合L1
'''
#ck：包含候选集合的列表，D:数据集，minSupport:感兴趣项集的最小支持度
def scanD(dataSet, Ck, minSupport):
    ssCnt = {}
    numItems = float(len( list(map(set, dataSet))))
    D = map(set, dataSet)


    for tid in D:                   #比较每个候选项集是否每个数据样本的子集，统计每个候选项集为数据样本子集的数目
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support     #  返回支持度的字典
    return retList, supportData

D = loadDataSet()
ck = createC1(D)
retList ,supportDate = scanD(D,ck,0.5)
print(retList)
print(supportDate)


