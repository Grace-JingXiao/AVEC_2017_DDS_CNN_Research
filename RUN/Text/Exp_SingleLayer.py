import os
import numpy
from Auxiliary.Loader import Loader_Text_Raw
from Model.Text.SingleLayer import SingleLayer
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer

if __name__ == '__main__':
    convSize = [2]
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = \
        Loader_Text_Raw(maxSentence=9999)

    attention = RNN_LocalAttentionInitializer
    attentionName = 'RLA'
    attentionScope = 1

    classifier = SingleLayer(trainData=trainData, trainLabel=trainLabel, attention=attention,
                             attentionName=attentionName, attentionScope=attentionScope, convSize=convSize)
    # classifier.Train(logName='log.csv')

    convName = str(convSize[0])
    for sample in convSize[1:]:
        convName += '-%d' % sample
    # classifier.Valid()

    savepath = 'D:/PythonProjects_Data/Exp-TXT/Single-CRNN-%s-%s' % (attentionName, convName)
    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' %
              (episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(savepath=savepath + '-TestResult/Result-%04d.csv' % episode, testData=testData,
                        testLabel=testLabel)
