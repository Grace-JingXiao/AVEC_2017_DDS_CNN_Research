import os
import numpy
from Model.SpeechRecognition.CNN_Seq2Seq import CNN_Seq2Seq
from Auxiliary.Loader import Loader_SpeechRecognition

if __name__ == '__main__':
    totalData, totalLabel, totalSeq = Loader_SpeechRecognition(maxSamples=99999)
    # print(numpy.max(numpy.max(totalLabel)))
    classifier = CNN_Seq2Seq(trainData=totalData, trainLabel=totalLabel, trainSeq=totalSeq, learningRate=1E-4,
                             batchSize=128)
    savepath = 'D:/PythonProjects_Data/AVEC_SR/'
    os.makedirs(savepath)

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' % (episode, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + 'Network-%04d' % episode)
