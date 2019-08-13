import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.DepressionRecognition.MultiCNN import SingleCNN

if __name__ == '__main__':
    convSize = 4
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_CNN(
        partName='CNN-10-Seq')
    classifier = SingleCNN(trainData=trainData, trainSeq=trainSeq, trainLabel=trainLabel, convSize=convSize)

    savepath = 'D:/PythonProjects_Data/Exp/CRNN-%d' % convSize
    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' %
              (episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(savepath=savepath + '-TestResult/Result-%04d.csv' % episode, testData=testData,
                        testLabel=testLabel, testSeq=testSeq)
