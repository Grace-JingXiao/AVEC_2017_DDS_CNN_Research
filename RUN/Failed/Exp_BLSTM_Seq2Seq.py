import os
import numpy
from Model.SpeechRecognition.BLSTM_Seq2Seq import BLSTM_Seq2Seq
from Auxiliary.Loader import Loader_SpeechRecognition

if __name__ == '__main__':
    totalData, totalLabel, totalSeq = Loader_SpeechRecognition()
    # print(numpy.max(numpy.max(totalLabel)))
    classifier = BLSTM_Seq2Seq(trainData=totalData, trainLabel=totalLabel, trainSeq=totalSeq, hiddenNoduleNumber=128)
    classifier.Train(logName=None)
