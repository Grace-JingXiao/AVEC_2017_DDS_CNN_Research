import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.DepressionRecognition.MultiCNN import SingleCNN
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer

if __name__ == '__main__':
    convSize = 4
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_CNN(
        partName='CNN-10-Seq')
    firstAttention = RNN_StandardAttentionInitializer
    firstAttentionName = 'RSA'
    firstAttentionScope = None
    secondAttention = RNN_StandardAttentionInitializer
    secondAttentionName = 'RSA'
    secondAttentionScope = None

    classifier = SingleCNN(
        trainData=trainData, trainSeq=trainSeq, trainLabel=trainLabel, convSize=convSize, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope, secondAttention=secondAttention,
        secondAttentionName=secondAttentionName, secondAttentionScope=secondAttentionScope)

    savepath = '/mnt/external/Bobs/AVEC2017/Exp/CRNN-%s-%s-%d' % (firstAttentionName, secondAttentionName, convSize)
    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' %
              (episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(savepath=savepath + '-TestResult/Result-%04d.csv' % episode, testData=testData,
                        testLabel=testLabel, testSeq=testSeq)
