from numpy import *
from tkinter import *
import regTrees

def reDraw(tolS,tolN):
    pass
def drawNewTree():
    pass

root = Tk()

# 标签部件
Label(root,text="Plot Place Holder").grid(row=0,columnspan = 3)
Label(root,text="tolN").grid(row = 1,column = 0)
# 文本输入部件，Entry部件是一个允许单行文本输入的文本框
tolNentry = Entry(root)
tolNentry.grid(row = 1,column = 1)
tolNentry.insert(0,'10')

Label(root,text="tolS").grid(row = 2,column = 0)
tolSentry = Entry(root)
tolSentry.grid(row=2,column = 1)
tolSentry.insert(0,'1.0')

# 按钮部件
Button(root,text="ReDraw",command = drawNewTree).grid(row = 1,column =2 ,rowspan =3)

# 复选按钮部件
chkBtnVar = IntVar()  # 按钮整数值，IntVar，新创建的变量
chkBtn = Checkbutton(root,text = "Model Tree",variable = chkBtnVar)
# columnspan,合并单元格
chkBtn.grid(row = 3,column = 0,columnspan =2)
reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)

reDraw(1.0,10)
root.mainloop()