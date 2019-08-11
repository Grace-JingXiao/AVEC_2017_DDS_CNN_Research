import numpy
import os


def Loader_CNN(partName, maxSentence=5):
    loadPath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/%s/' % partName
    labelPath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/'

    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = [], [], [], [], [], [], [], [], []

    for loadPart in ['train', 'dev', 'test']:
        labelData = numpy.genfromtxt(fname=os.path.join(labelPath, '%sLabel.csv' % loadPart), dtype=int,
                                     delimiter=',')[1:]
        for searchIndex in range(min(len(labelData), maxSentence)):
            batchData = numpy.load(file=os.path.join(loadPath, loadPart, '%d_P-Data.npy' % labelData[searchIndex][0]))
            batchSeqCurrent = numpy.load(
                file=os.path.join(loadPath, loadPart, '%d_P-Seq.npy' % labelData[searchIndex][0]))

            batchSeq = []
            for sample in batchSeqCurrent: batchSeq.append(int(sample / 2))
            print('Loading', loadPart, labelData[searchIndex][0], numpy.shape(batchData))

            if loadPart == 'train':
                trainData.append(batchData)
                trainLabel.append(labelData[searchIndex][2])
                trainSeq.append(batchSeq)
            if loadPart == 'dev':
                developData.append(batchData)
                developLabel.append(labelData[searchIndex][2])
                developSeq.append(batchSeq)
            if loadPart == 'test':
                testData.append(batchData)
                testLabel.append(labelData[searchIndex][2])
                testSeq.append(batchSeq)

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(developData), numpy.shape(developLabel), numpy.shape(developSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq


def Loader_SpeechRecognition(maxSamples=1):
    loadpath = 'D:/PythonProjects_Data/Data_SpeechRecognition/'

    totalData, totalLabel, totalSeq = [], [], []

    for fold in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, fold))[0:maxSamples]:
            if filename.find('npy') == -1: continue

            partData = numpy.load(os.path.join(loadpath, fold, filename)) * 10
            currentSeq = numpy.genfromtxt(fname=os.path.join(loadpath, fold, filename.replace('Data.npy', 'Seq.csv')),
                                          dtype=int, delimiter=',')
            partSeq = []
            for sample in currentSeq:
                partSeq.append(int(sample / 4))

            with open(os.path.join(loadpath, fold, filename.replace('Data.npy', 'Label.csv'))) as file:
                labelCurrent = file.readlines()

            partLabel = []
            for sample in labelCurrent:
                sample = sample.split(',')[0:-1]

                result = []
                for subsample in sample:
                    result.append(int(subsample))
                partLabel.append(result)

            print(fold, filename, numpy.shape(partData), numpy.shape(partSeq), numpy.shape(partLabel))
            totalData.extend(partData)
            totalSeq.extend(partSeq)
            totalLabel.extend(partLabel)

    print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.shape(totalSeq))
    return totalData, totalLabel, totalSeq


if __name__ == '__main__':
    Loader_CNN(partName='CNN-10-Seq', maxSentence=9999)
