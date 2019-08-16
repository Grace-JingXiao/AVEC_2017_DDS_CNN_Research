import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.AutoEncoder.AutoEncoder_Conv2D import AutoEncoder_Conv2D

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_CNN(
        partName='CNN-10-Seq')
    totalData = []
    for treat in [trainData, developData, testData]:
        for sample in treat: totalData.extend(sample)
    print(numpy.shape(totalData))

    classifier = AutoEncoder_Conv2D(trainData=totalData, learningRate=1E-4)
    savepath = 'D:/PythonProjects_Data/Exp/Conv2D/'
    os.makedirs(savepath)

    for episode in range(100):
        print('Episode %d Total Loss = %f' % (episode, classifier.Train(savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + 'Network-%04d' % episode)
