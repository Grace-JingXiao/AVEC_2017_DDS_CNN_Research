import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/AutoEncoder/'
    savepath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/AutoEncoder-Normalization/'
    # os.makedirs(savepath)

    trainData = numpy.genfromtxt(fname=os.path.join(loadpath, 'Conv3D-TrainData.csv'), dtype=float, delimiter=',')
    developData = numpy.genfromtxt(fname=os.path.join(loadpath, 'Conv3D-DevelopData.csv'), dtype=float, delimiter=',')
    testData = numpy.genfromtxt(fname=os.path.join(loadpath, 'Conv3D-TestData.csv'), dtype=float, delimiter=',')

    totalData = numpy.concatenate([trainData, developData, testData], axis=0)
    totalData = scale(totalData)

    print(numpy.shape(trainData), numpy.shape(developData), numpy.shape(testData), numpy.shape(totalData))

    with open(os.path.join(savepath, 'Conv3D-TrainData.csv'), 'w') as file:
        writeData = totalData[0:numpy.shape(trainData)[0]]
        for indexX in range(numpy.shape(writeData)[0]):
            for indexY in range(numpy.shape(writeData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(writeData[indexX][indexY]))
            file.write('\n')
        print(numpy.shape(writeData))
    with open(os.path.join(savepath, 'Conv3D-DevelopData.csv'), 'w') as file:
        writeData = totalData[numpy.shape(trainData)[0]:numpy.shape(trainData)[0] + numpy.shape(developData)[0]]
        for indexX in range(numpy.shape(writeData)[0]):
            for indexY in range(numpy.shape(writeData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(writeData[indexX][indexY]))
            file.write('\n')
        print(numpy.shape(writeData))
    with open(os.path.join(savepath, 'Conv3D-TestData.csv'), 'w') as file:
        writeData = totalData[numpy.shape(trainData)[0] + numpy.shape(developData)[0]:]
        for indexX in range(numpy.shape(writeData)[0]):
            for indexY in range(numpy.shape(writeData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(writeData[indexX][indexY]))
            file.write('\n')
        print(numpy.shape(writeData))
