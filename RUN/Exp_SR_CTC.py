from Model.SpeechRecognition.CNN_CTC import CNN_CTC
from Auxiliary.Loader import Loader_SpeechRecognition
from Model.AttentionMechanism.CNN_M2M_StandardAttention import CNN_M2M_StandardAttention_Initializer

if __name__ == '__main__':
    totalData, totalLabel, totalSeq = Loader_SpeechRecognition()
    classifier = CNN_CTC(
        trainData=totalData, trainLabel=totalLabel, trainSeq=totalSeq,
        M2M_Attention=CNN_M2M_StandardAttention_Initializer, M2M_AttentionName='CSA')
    classifier.Valid()
