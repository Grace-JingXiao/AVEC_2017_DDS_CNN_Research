import os
import numpy
import tensorflow
from Auxiliary.Loader import Loader_CNN
from Model.DepressionRecognition.MultiCNNwithSR import MultiCNNWithSR
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer

if __name__ == '__main__':
    convSize = [2, 3, 4]
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_CNN(
        partName='CNN-10-Seq', maxSentence=9999)

    firstAttention = RNN_StandardAttentionInitializer
    firstAttentionName = 'RSA'
    firstAttentionScope = None
    secondAttention = RNN_StandardAttentionInitializer
    secondAttentionName = 'RSA'
    secondAttentionScope = None
    weight = 1

    loadpath = 'D:/PythonProjects_Data/Exp-SR/MultiCNN_%s/Network-0069' % firstAttentionName

    classifier = MultiCNNWithSR(
        trainData=trainData, trainSeq=trainSeq, trainLabel=trainLabel, transformWeight=weight, convSize=convSize,
        firstAttention=firstAttention, firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope,
        secondAttention=secondAttention, secondAttentionName=secondAttentionName,
        secondAttentionScope=secondAttentionScope)
    convName = str(convSize[0])
    for sample in convSize[1:]:
        convName += '-%d' % sample

    savepath = 'D:/PythonProjects_Data/Exp-DR-with-SR/Multi-CRNN-%s-%s-%s-%d' % (
        firstAttentionName, secondAttentionName, convName, weight)
    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' %
              (episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(savepath=savepath + '-TestResult/Result-%04d.csv' % episode, testData=testData,
                        testLabel=testLabel, testSeq=testSeq)
