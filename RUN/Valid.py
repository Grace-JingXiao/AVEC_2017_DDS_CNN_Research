import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.DepressionRecognition.MultiCNN import SingleCNN
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer

if __name__ == '__main__':
    convSize = 4
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = Loader_CNN(
        partName='CNN-10-Seq')
    firstAttention = RNN_LocalAttentionInitializer
    firstAttentionName = 'RLA'
    firstAttentionScope = 1
    secondAttention = RNN_LocalAttentionInitializer
    secondAttentionName = 'RLA'
    secondAttentionScope = 1

    classifier = SingleCNN(
        trainData=trainData, trainSeq=trainSeq, trainLabel=trainLabel, convSize=convSize, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope, secondAttention=secondAttention,
        secondAttentionName=secondAttentionName, secondAttentionScope=secondAttentionScope)
    classifier.Valid()
