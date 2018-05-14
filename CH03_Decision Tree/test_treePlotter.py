#“- * - coding: utf - 8 -*-”
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle ='sawtooth',fc = '0.8')
leafNode = dict(boxstyle = 'round4',fc = '0.8')
arrow_args = dict(arrowstyle = '<-')

def plotNode(nodeTxt,centerPr,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt ,xy = parentPt ,xycoords = 'axes fraction',xytex=centerPr,
                            textcoords='axes fraction',va ='center',ha='nodeType',
                            bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot().ax1 = plt.subplot(111,frameon = False)
    plotNode(u'决策点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(u'叶节点',(0.8,0.1),(0.3,0.8),leafNode )
    plt.show()

