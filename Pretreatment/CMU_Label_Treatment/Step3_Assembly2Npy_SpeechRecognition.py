import numpy
import os

MAX_LEN = 1000

if __name__ == '__main__':
    datapath = 'D:/PythonProjects_Data/AVEC2017_Data/Step4_Normalization_Part2/'
    labelpath = 'D:/PythonProjects_Data/AVEC2017_Data/CMU_Step2_Label/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Data/Data_SpeechRecognition/'

    for foldX in os.listdir(datapath)[2:3]:
        # os.makedirs(os.path.join(savepath, foldX))
        for foldY in os.listdir(os.path.join(datapath, foldX))[100:]:
            totalData, totalLabel, totalSeq = [], [], []

            for filename in os.listdir(os.path.join(datapath, foldX, foldY)):
                data = numpy.genfromtxt(fname=os.path.join(datapath, foldX, foldY, filename), dtype=float,
                                        delimiter=',')
                seq = min(numpy.shape(data)[0], MAX_LEN)
                label = numpy.genfromtxt(fname=os.path.join(labelpath, foldX, foldY, filename), dtype=int,
                                         delimiter=',')
                if (len(numpy.shape(data)) < 2) or (numpy.shape(label)[0] == 0) or \
                        (numpy.shape(data)[0] < numpy.shape(label)[0] * 4): continue

                #########################################################################

                if numpy.shape(data)[0] < MAX_LEN:
                    data = numpy.concatenate(
                        [data, numpy.zeros([MAX_LEN - numpy.shape(data)[0], numpy.shape(data)[1]])], axis=0)
                else:
                    data = data[0:MAX_LEN]

                #########################################################################

                totalData.append(data)
                totalSeq.append(seq)
                totalLabel.append(label)
                # print(foldX, foldY, filename, numpy.shape(data), seq, label)
            print(foldX, foldY, numpy.shape(totalData), numpy.shape(totalSeq), numpy.shape(totalLabel))
            numpy.save(file=os.path.join(savepath, foldX, foldY + '-Data.npy'), arr=totalData)
            with open(os.path.join(savepath, foldX, foldY + '-Seq.csv'), 'w') as file:
                for index in range(numpy.shape(totalSeq)[0]):
                    if index != 0: file.write(',')
                    file.write(str(totalSeq[index]))

            with open(os.path.join(savepath, foldX, foldY + '-Label.csv'), 'w') as file:
                for indexX in range(numpy.shape(totalLabel)[0]):
                    for indexY in range(numpy.shape(totalLabel[indexX])[0]):
                        if indexY != 0: file.write(',')
                        file.write(str(totalLabel[indexX][indexY]))
                    file.write('\n')
            # exit()
