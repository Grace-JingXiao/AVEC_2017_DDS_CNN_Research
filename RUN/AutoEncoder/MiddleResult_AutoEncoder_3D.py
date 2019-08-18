import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.AutoEncoder.AutoEncoder_Conv3D import AutoEncoder_Conv3D

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_CNN(
        partName='CNN-10-Seq', maxSentence=9999)

    classifier = AutoEncoder_Conv3D(
        trainData=numpy.concatenate([trainData, developData, testData], axis=0), learningRate=1E-3)
    classifier.Load(r'D:\PythonProjects_Data\Conv3D-Huber\Network-4674')
    classifier.MiddleResultGenerate(savepath='Conv3D-TrainData.csv', testData=trainData)
    classifier.MiddleResultGenerate(savepath='Conv3D-DevelopData.csv', testData=developData)
    classifier.MiddleResultGenerate(savepath='Conv3D-TestData.csv', testData=testData)
