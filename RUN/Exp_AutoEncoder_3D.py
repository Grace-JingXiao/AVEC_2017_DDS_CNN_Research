import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.AutoEncoder.AutoEncoder_Conv3D import AutoEncoder_Conv3D

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_CNN(
        partName='CNN-10-Seq')

    classifier = AutoEncoder_Conv3D(
        trainData=numpy.concatenate([trainData, developData, testData], axis=0)[0:1], learningRate=1E-3)
    savepath = 'D:/PythonProjects_Data/Exp/Conv3D/'
    os.makedirs(savepath)

    for episode in range(100):
        print('Episode %d Total Loss = %f' % (episode, classifier.Train(savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + 'Network-%04d' % episode)
