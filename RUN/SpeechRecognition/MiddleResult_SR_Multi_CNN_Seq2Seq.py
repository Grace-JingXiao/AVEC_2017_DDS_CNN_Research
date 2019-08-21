import os
from Auxiliary.Loader import Loader_SpeechRecognition
from Model.SpeechRecognition.MultiCNN_Seq2Seq import MultiCNN_Seq2Seq
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer

if __name__ == '__main__':
    totalData, totalLabel, totalSeq = Loader_SpeechRecognition()
    # print(numpy.max(numpy.max(totalLabel)))

    attention = RNN_StandardAttentionInitializer
    attentionName = 'RSA'
    attentionScope = None
    classifier = MultiCNN_Seq2Seq(
        trainData=totalData, trainLabel=totalLabel, trainSeq=totalSeq, attention=attention, attentionName=attentionName,
        attentionScope=attentionScope, convSize=[2, 3, 4], learningRate=1E-4, batchSize=128)
    classifier.MiddleResultGeneration(testData=totalData, testLabel=totalLabel, testSeq=totalSeq)
