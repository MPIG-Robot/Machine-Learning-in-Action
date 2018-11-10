from numpy import *
import regTrees
testMat = mat(eye(4))
mat1 = regTrees.binSplitDataSet(testMat,1,0.5)
# print(testMat)
mat0 = testMat[nonzero(testMat[:, 1] > 0.5)[0], :]
# print(mat0)
print(mat1)