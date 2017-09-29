from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy import *

##KNN example

##Get Data from file
def getMatrix(filename):
    fr=open(filename)
    arrayOfLines=fr.readlines()
    numberOfLines=len(arrayOfLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOfLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

##draw 3d
def PrintFigure(datingDataMat,datingLabels):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    num=len(datingDataMat)
    for i in range(num):
        if datingLabels[i]==1:
            ax.scatter(datingDataMat[i][0],datingDataMat[i][1],datingDataMat[i][2],c='b',marker='x')
        elif datingLabels[i]==2:
            ax.scatter(datingDataMat[i][0],datingDataMat[i][1],datingDataMat[i][2],c='r',marker='o')
        elif datingLabels[i]==3:
            ax.scatter(datingDataMat[i][0],datingDataMat[i][1],datingDataMat[i][2],c='g',marker='*')
        elif datingLabels[i]==0:
            ax.scatter(datingDataMat[i][0],datingDataMat[i][1],datingDataMat[i][2],marker='1')
    plt.show()
        

##draw 3d
##datingDataMat,datingLabels=getMatrix('./knnTest.txt')
##PrintFigure(datingDataMat,datingLabels)

##data normalization processing
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

##knn classify
def classify(inx,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inx,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    import operator
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

hoRatio=0.20
datingDataMat,datingLabels=getMatrix('./knnTest.txt')
normMat,ranges,minVals=autoNorm(datingDataMat)
#PrintFigure(normMat,datingLabels)
m=normMat.shape[0]
numTestVecs=int(m*hoRatio)
errorcount=0.0
for i in range(numTestVecs):
    classifierResult=classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
    print("Rt is %d, Real is %d"%(classifierResult,datingLabels[i]))
    if(classifierResult!=datingLabels[i]):
        errorcount+=1.0
    print("%f"%(errorcount/float(numTestVecs)))
          
