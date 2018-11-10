from numpy import *
from tkinter import *
import regTrees
import matplotlib
# TKagg ：后端，用于实现绘图和不同应用之间接口
matplotlib.use('TkAgg')
# 声明将TKagg和Matplotlib链接起来
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 以用户输入的终止条件来绘图
def reDraw (tols,tolN):
    # 清空图像，重新绘图
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    # 判断复选框是否选中
    if chkBtnVar.get():
        if tolN<2:
            tolN=2
        myTree=regTrees.createTree(reDraw.rawDat,regTrees.modelLeaf,regTrees.modelErr,(tols,tolN))
        yHat=regTrees.createForeCast(myTree,reDraw.testDat,regTrees.modelTreeEval)
    else:
        myTree=regTrees.createTree(reDraw.rawDat,ops=(tols,tolN))
        yHat=regTrees.createForeCast(myTree,reDraw.testDat)
    # 绘出真实值
    reDraw.a.scatter(reDraw.rawDat[:,0].A,reDraw.rawDat[:,1].A,s=5)
    # 会出预测值
    reDraw.a.plot(reDraw.testDat,yHat,linewidth=2.0)
    reDraw.canvas.show()

# 从文本输入框中获取树创建终止条件，没有则用默认值
# 如果python可以把输入文本解析成整数就继续执行，如果不能识别则输出错误信息，，清空并恢复默认值
def getInputs():
    try:tolN=int(tolNentry.get())
    except:
        tolN=10
        print("enter Integer for tolN")
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:tolS=float(tolSentry.get())
    except:
        tolS=1.0
        print("enter Float for tolS")
        tolSentry.delete(0,END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS=getInputs()
    reDraw(tolS,tolN)

root=Tk()
# Tkagg在所选GUI框架上调用Agg，把Agg呈现在画布上
reDraw.f=Figure(figsize=(5,4),dpi=100)
reDraw.canvas=FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)

Label(root,text="tolN").grid(row=1,column=0)
tolNentry=Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')

Label(root,text="tolS").grid(row=2,column=0)
tolSentry=Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1.0')

Button(root,text="ReDraw",command=drawNewTree).grid(row=1,column=2,rowspan=3)
chkBtnVar=IntVar()
chkBtn=Checkbutton(root,text="Model Tree",variable=chkBtnVar)
chkBtn.grid(row=3,column=0,columnspan=2)

reDraw.rawDat=mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat=arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0,10)
root.mainloop()