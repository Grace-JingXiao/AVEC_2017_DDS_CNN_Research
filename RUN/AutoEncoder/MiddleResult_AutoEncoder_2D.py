import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.AutoEncoder.AutoEncoder_Conv2D import AutoEncoder_Conv2D

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = \
        Loader_CNN(partName='CNN-10-Seq', maxSentence=9999)

    classifier = AutoEncoder_Conv2D(trainData=trainData, learningRate=5E-4, batchSize=64)
    loadpath = 'D:/PythonProjects_Data/Conv2D-Huber/Network-0099'

    classifier.Load(loadpath=loadpath)
    os.makedirs('Conv2D-TrainData')
    os.makedirs('Conv2D-DevelopData')
    os.makedirs('Conv2D-TestData')
    classifier.MiddleResultGenerate(savepath='Conv2D-TrainData/', testData=trainData)
    classifier.MiddleResultGenerate(savepath='Conv2D-DevelopData/', testData=developData)
    classifier.MiddleResultGenerate(savepath='Conv2D-TestData/', testData=testData)
