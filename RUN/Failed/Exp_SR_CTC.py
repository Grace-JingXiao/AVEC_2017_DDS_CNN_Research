from Model.SpeechRecognition.CNN_CTC import CNN_CTC
from Auxiliary.Loader import Loader_SpeechRecognition

if __name__ == '__main__':
    totalData, totalLabel, totalSeq = Loader_SpeechRecognition()
    classifier = CNN_CTC(
        trainData=totalData, trainLabel=totalLabel, trainSeq=totalSeq)
    # print(totalSeq)
    # classifier.Valid()

    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' % (episode, classifier.Train()))
