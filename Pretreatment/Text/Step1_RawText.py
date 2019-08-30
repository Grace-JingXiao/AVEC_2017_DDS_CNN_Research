import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Step2_VoiceSeparate/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Data/Text_Step1_RawText/'

    for foldName in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, foldName))
        for partName in os.listdir(os.path.join(loadpath, foldName)):
            with open(os.path.join(savepath, foldName, partName + '.csv'), 'w') as file:
                rawData = numpy.genfromtxt(
                    fname=os.path.join(loadpath, foldName, partName, 'Transcription.csv'), dtype=str, delimiter=',')
                for sample in rawData[1:]:
                    if sample[3] != 'Participant': continue
                    file.write(sample[4] + '\n')
            # exit()
