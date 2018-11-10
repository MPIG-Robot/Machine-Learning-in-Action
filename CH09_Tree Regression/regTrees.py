from numpy import *
import matplotlib.pyplot as plt

# 解析样本数据
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每行映射成浮点数
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

# 拆分数据集函数，数组过滤方式
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

# 回归树的切分函数
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    # tolS：容许的误差下降值，tolN：切分的最小样本数
    tolS=ops[0];tolN=ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:    # 1
        return None,leafType(dataSet)
    m,n=shape(dataSet)
    S=errType(dataSet)
    # 初始化最小误差，最佳切分特征索引，最佳切分特征值
    bestS=inf;bestIndex=0;bestValue=0
    for featIndex in range (n-1):
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S-bestS)<tolS:    # 2
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN):    # 3
        return None,leafType(dataSet)
    return bestIndex,bestValue

# 树构建函数
def createTree(dataSet,leafType = regLeaf,errType = regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:
        return val
    retTree = { }
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree

# if __name__=='__main__':
#     myDat = loadDataSet('ex00.txt')
#     # myDat = loadDataSet('ex0.txt')
#     myDat = mat(myDat)
#     tree = createTree(myDat)
#     print(tree)
#     plt.plot(myDat[:,0],myDat[:,1],'ro')
#     # plt.plot(myDat[:, 1], myDat[:, 2], 'ro')
#     plt.show()

# 回归树剪枝函数
# 测试输入变量是否是一棵树，返回布尔型结果
def isTree(obj):
    return (type(obj).__name__=='dict')

# 递归函数
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    # 递归直至找到两个叶节点，返回二者的均值
    return (tree['left']+tree['right'])/2.0

#入口参数：待剪枝的树与剪枝所需要的测试数据
def prune(tree,testData):
    # 若测试集为空，则对树进行塌陷处理
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right'])or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree (tree['left'])and not isTree (tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1]-tree['left'],2))+sum(power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

# if __name__ =='__main__':
#     myDat = loadDataSet('ex2.txt')
#     myDat = mat(myDat)
#     myTree = createTree(myDat,ops = (0,1))
#     myDatTest = loadDataSet('ex2test.txt')
#     myDatTest = mat(myDatTest)
#     tree = prune(myTree,myDatTest)
#     print(tree)
#     plt.plot(myDat[:, 0], myDat[:, 1], 'ro')
#     plt.show()

# 模型树的叶节点生成函数
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    # 将数据集中第一列初始化为1，格式化为X，Y
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    # 判断矩阵是否可逆
    if linalg.det(xTx) == 0.0:
        # 该矩阵是奇异矩阵，不能取反，尝试增加ops的第二个参数
        raise NameError('This matrix is singular,cannot do inverse,try increasing the second value of ops')
    ws = xTx.I*(X.T*Y)  #根据预测值和真值的平方误差，求导得到
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y-yHat,2))

# if __name__ == '__main__':
#     myDat = loadDataSet('ex00.txt')
#     myMat = mat(myDat)
#     tree4 = createTree(myMat,modelLeaf,modelErr,ops = (1,10))
#     print(tree4)
#     plt.plot(myMat[:, 0], myMat[:, 1], 'ro')
#     plt.show()

# 用树回归进行预测
# 回归树的叶节点为float型常量
def regTreeEval(model,inDat):
    return float(model)

# 模型树的叶节点浮点型参数的线性方程
def modelTreeEval(model,inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    # 返回浮点型的回归系数向量
    return float(X*model)

# 树预测
# tree；树回归模型
# inData：输入数据
# modelEval：叶节点生成类型，需指定，默认回归树类型
def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval = regTreeEval):
    m = len(testData)
    yHat = mat (zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat

# if __name__=='__main__':
#     trainMat = mat(loadDataSet('bikeSpeedVsiq_train.txt'))
#     testMat = mat(loadDataSet('bikeSpeedVsiq_test.txt'))
#     # 回归树
#     myTree = createTree(trainMat,ops = (1,20))
#     yHat = createForeCast(myTree,testMat[:,0])
#     print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
#
#     # 模型树
#     myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
#     yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
#     print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
#
#     ws, X, Y = linearSolve(trainMat)
#     # print(ws)
#     for i in range(shape(testMat)[0]):
#         yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
#     print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])