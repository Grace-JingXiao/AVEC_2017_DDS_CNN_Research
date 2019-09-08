import os
import numpy
from Auxiliary.Loader import Loader_Audio
from Model.Audio.SingleFeaturesSingleCNN import SingleCNN_Audio
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_Audio(
        partName='AUs', maxSentence=99999)
    firstAttention = RNN_StandardAttentionInitializer
    firstAttentionName = 'RSA'
    firstAttentionScope = None
    secondAttention = RNN_StandardAttentionInitializer
    secondAttentionName = 'RSA'
    secondAttentionScope = None

    savepath = 'D:/PythonProjects_Data/Exp_Audio/Single_%s_%s' % (firstAttentionName, secondAttentionName)

    classifier = SingleCNN_Audio(
        trainData=trainData, trainSeq=trainSeq, trainLabel=trainLabel, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope, secondAttention=secondAttention,
        secondAttentionName=secondAttentionName, secondAttentionScope=secondAttentionScope)

    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' %
              (episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(savepath=savepath + '-TestResult/Result-%04d.csv' % episode, testData=testData,
                        testLabel=testLabel, testSeq=testSeq)
