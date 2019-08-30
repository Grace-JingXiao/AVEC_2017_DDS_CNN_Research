import os
import numpy
from Auxiliary.Loader import Loader_Text
from Model.Text.MultiCNN_Text import MultiCNN_Text
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer

if __name__ == '__main__':
    convSize = [2, 3, 4]
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = \
        Loader_Text(maxSentence=9999)
    firstAttention = RNN_StandardAttentionInitializer
    firstAttentionName = 'RSA'
    firstAttentionScope = None
    secondAttention = RNN_StandardAttentionInitializer
    secondAttentionName = 'RSA'
    secondAttentionScope = None

    classifier = MultiCNN_Text(
        trainData=trainData, trainSeq=trainSeq, trainLabel=trainLabel, convSize=convSize, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope, secondAttention=secondAttention,
        secondAttentionName=secondAttentionName, secondAttentionScope=secondAttentionScope)
    convName = str(convSize[0])
    for sample in convSize[1:]:
        convName += '-%d' % sample
    # classifier.Valid()

    savepath = 'D:/PythonProjects_Data/Exp-TXT/Multi-CRNN-%s-%s-%s' % (
        firstAttentionName, secondAttentionName, convName)
    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' %
              (episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(savepath=savepath + '-TestResult/Result-%04d.csv' % episode, testData=testData,
                        testLabel=testLabel, testSeq=testSeq)
