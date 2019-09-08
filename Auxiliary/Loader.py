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


def Loader_AutoEncoderResult(maxSentence=5):
    loadpath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/AutoEncoder-Normalization/'
    trainDataSentence, developDataSentence, testDataSentence = [], [], []
    trainDataInterview = numpy.genfromtxt(
        fname=os.path.join(loadpath, 'Conv3D-TrainData.csv'), dtype=float, delimiter=',')[0:maxSentence]
    developDataInterview = numpy.genfromtxt(
        fname=os.path.join(loadpath, 'Conv3D-DevelopData.csv'), dtype=float, delimiter=',')[0:maxSentence]
    testDataInterview = numpy.genfromtxt(
        fname=os.path.join(loadpath, 'Conv3D-TestData.csv'), dtype=float, delimiter=',')[0:maxSentence]

    for foldname in ['Conv2D-TrainData', 'Conv2D-DevelopData', 'Conv2D-TestData']:
        for index in range(maxSentence):
            filename = '%04d.csv' % index
            if not os.path.exists(os.path.join(loadpath, foldname, filename)): continue
            print('Loading', foldname, filename)
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            if foldname.find('TrainData') != -1: trainDataSentence.append(data)
            if foldname.find('DevelopData') != -1: developDataSentence.append(data)
            if foldname.find('TestData') != -1: testDataSentence.append(data)

    print(numpy.shape(trainDataSentence), numpy.shape(developDataSentence), numpy.shape(testDataSentence))
    print(numpy.shape(trainDataInterview), numpy.shape(developDataInterview), numpy.shape(testDataInterview))
    return trainDataSentence, developDataSentence, testDataSentence, trainDataInterview, developDataInterview, testDataInterview


def Loader_Text(maxSentence=5):
    loadPath = 'D:/PythonProjects_Data/Data_Text/'
    labelPath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/'

    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = [], [], [], [], [], [], [], [], []

    for loadPart in ['train', 'dev', 'test']:
        labelData = numpy.genfromtxt(fname=os.path.join(labelPath, '%sLabel.csv' % loadPart), dtype=int,
                                     delimiter=',')[1:]
        for searchIndex in range(min(len(labelData), maxSentence)):
            batchData = numpy.genfromtxt(
                fname=os.path.join(loadPath, loadPart, '%d_P.csv' % labelData[searchIndex][0]), dtype=float,
                delimiter=',')
            batchSeq = numpy.genfromtxt(
                fname=os.path.join(loadPath, loadPart, '%d_P_Seq.csv' % labelData[searchIndex][0]), dtype=float,
                delimiter=',')

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


def Loader_Text_Raw(maxSentence=5):
    loadPath = 'D:/PythonProjects_Data/Data_Text_Raw/'
    labelPath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/'

    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, \
    testData, testLabel, testSeq = [], [], [], [], [], [], [], [], []

    for loadPart in ['train', 'dev', 'test']:
        labelData = numpy.genfromtxt(fname=os.path.join(labelPath, '%sLabel.csv' % loadPart), dtype=int,
                                     delimiter=',')[1:]
        for searchIndex in range(min(len(labelData), maxSentence)):
            with open(os.path.join(loadPath, loadPart, '%d_P.csv' % labelData[searchIndex][0]), 'r') as file:
                batchData = file.read()
                batchData = batchData.replace('\n', '')[0:-1]
            # print(batchData)

            current = []
            for sample in batchData.split(','):
                current.append(int(sample))

            if loadPart == 'train':
                trainData.append(current)
                trainLabel.append(labelData[searchIndex][2])
                trainSeq.append(len(current))
            if loadPart == 'dev':
                developData.append(current)
                developLabel.append(labelData[searchIndex][2])
                developSeq.append(len(current))
            if loadPart == 'test':
                testData.append(current)
                testLabel.append(labelData[searchIndex][2])
                testSeq.append(len(current))

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print(numpy.shape(developData), numpy.shape(developLabel), numpy.shape(developSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq


def Loader_Audio(partName, maxSentence=5):
    loadPath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Data_%s/' % partName
    labelPath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/'

    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = [], [], [], [], [], [], [], [], []

    for loadPart in ['train', 'dev', 'test']:
        labelData = numpy.genfromtxt(fname=os.path.join(labelPath, '%sLabel.csv' % loadPart), dtype=int,
                                     delimiter=',')[1:]
        for searchIndex in range(min(len(labelData), maxSentence)):
            batchData = numpy.load(file=os.path.join(loadPath, '%d_P.npy' % labelData[searchIndex][0]))
            batchSeq = numpy.load(file=os.path.join(loadPath, '%d_P_Seq.npy' % labelData[searchIndex][0]))

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

if __name__ == '__main__':
    Loader_Audio(partName='AUs')
