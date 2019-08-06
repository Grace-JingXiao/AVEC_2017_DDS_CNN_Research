import os
from Model.SpeechRecognition.CTC_Test import CTC
from Auxiliary.Loader import Loader_SpeechRecognition

if __name__ == '__main__':
    totalData, totalLabel, totalSeq = Loader_SpeechRecognition()
    classifier = CTC(trainData=totalData, trainLabel=totalLabel, trainSeqLength=totalSeq, featureShape=40, numClass=41,
                     batchSize=32)
    # classifier.Valid()
    savepath = 'D:/PythonProjects_Data/SimpleCTC/'
    os.makedirs(savepath)
    for episode in range(100):
        print('\nTrain Episode %d Total Loss = %f' % (episode, classifier.Train(logName=None)))
        classifier.Save(savepath=savepath + 'Network-%04d' % episode)
