
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    C1 = [frozenset(i) for i in C1]
    return  C1

def scanD(dataSet, Ck, minSupport):
    ssCnt = {}
    numItems = float(len( list(map(set, dataSet))))
    D = map(set, dataSet)
    for tid in D:
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
        supportData[key] = support
    return retList, supportData
'''
   前K-2个项相同时，将两个集合合并
'''
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # 前K-2项集相同的时候，
                retList.append(Lk[i] | Lk[j])  # 集合的合并
    return retList

def apriori(dataSet, minSupport):
    C1 = createC1(dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(dataSet, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

D = loadDataSet()
L , supportDate = apriori(D ,0.1)
print(L)
print(supportDate)



