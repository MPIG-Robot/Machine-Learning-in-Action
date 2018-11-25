
DataSet = [line.split() for line in open('mushroom.dat').readlines()]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    C1 = [frozenset(i) for i in C1]
    return  C1

def scanD_1(dataSet, Ck, minSupport):
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
    rList = []
    sData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        sData[key] = support
        if support >= minSupport:
            rList.insert(0, key)
        #sData[key] = support
    return rList, sData

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
    Lk = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        supportData[key] = support
        if support >= minSupport:
            Lk.insert(0,key)
            if key.intersection('2'):
                retList.insert(0, key)
        #supportData[key] = support
    return retList,Lk, supportData

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

def getC1(L1):
    List = []
    for item in L1:
        if item.intersection('2'):
            List.append(item)
    return List

def apriori(dataSet, minSupport):
    C1 = createC1(dataSet)
    L1, supportData = scanD_1(dataSet, C1, minSupport)
    List = getC1(L1)     #   首次不含2
    L = [L1]             #  首次全部
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk_2,Lk,supK = scanD(dataSet, Ck, minSupport)
        L.append(Lk)
        List.append(Lk_2)
        supportData.update(supK)
        k += 1
    return List, supportData

List , supportDate = apriori(DataSet ,0.45)
print('频繁项集:',List)
#print('L的支持列表:',supportDate)


def calcConf(freqSet, H, supportData, brl, minConf):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  #  需要合并的
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def generateRules(L, supportData, minConf):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):     #  H1 里不是单一的项
                H1 = calcConf(freqSet, H1, supportData, bigRuleList, minConf)
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:           # H1 里面是单一的项
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

print('关联规则:--------------------')
generateRules(List,supportDate,0.8)










