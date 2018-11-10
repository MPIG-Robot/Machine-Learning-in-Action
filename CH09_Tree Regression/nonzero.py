from numpy import *
def binSpiltDataSet(feature,value):
    dataSet=mat([[1,2,1,2],
              [1,3,3,3],
              [1,2,2,4],
              [1,1,1,1]])
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0,mat1
mat0,mat1=binSpiltDataSet(2,1)
print(mat0)
print(mat1)