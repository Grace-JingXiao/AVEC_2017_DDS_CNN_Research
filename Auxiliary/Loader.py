import numpy
import os


def Loader_SpeechRecognition(maxSamples=1):
    loadpath = 'D:/PythonProjects_Data/Data_SpeechRecognition/'

    totalData, totalLabel, totalSeq = [], [], []

    for fold in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, fold))[0:maxSamples]:
            if filename.find('npy') == -1: continue

            partData = numpy.load(os.path.join(loadpath, fold, filename))
            partSeq = numpy.genfromtxt(fname=os.path.join(loadpath, fold, filename.replace('Data.npy', 'Seq.csv')),
                                       dtype=int, delimiter=',')

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
    Loader_SpeechRecognition()
