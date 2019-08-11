import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.DepressionRecognition.MultiCNN import SingleCNN

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_CNN(
        partName='CNN-10-Seq')
    classifier = SingleCNN(trainData=trainData, trainSeq=trainSeq, trainLabel=trainLabel)

    savepath = 'D:/PythonProjects_Data/Exp/CRNN-2nd/'
    os.makedirs(savepath)

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' %
              (episode, classifier.Train(logName=savepath + 'Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + 'Network-%04d' % episode)
