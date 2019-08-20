import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/AutoEncoder/'
    savepath = 'D:/PythonProjects_Data/Data_AVEC2017_CNN/AutoEncoder-Normalization/'

    totalData = []

    for foldname in os.listdir(loadpath):
        if not os.path.isdir(os.path.join(loadpath, foldname)): continue
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            totalData.extend(data)
            print(foldname, filename, numpy.shape(totalData))
    print(numpy.shape(totalData))

    totalData = scale(totalData)

    startPosition = 0
    for foldname in os.listdir(loadpath):
        if not os.path.isdir(os.path.join(loadpath, foldname)): continue
        os.makedirs(os.path.join(savepath, foldname))

        for filename in os.listdir(os.path.join(loadpath, foldname)):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')

            writeData = totalData[startPosition:startPosition + numpy.shape(data)[0]]
            print(foldname, filename, startPosition, numpy.shape(writeData))
            with open(os.path.join(savepath, foldname, filename), 'w') as file:
                for indexX in range(numpy.shape(writeData)[0]):
                    for indexY in range(numpy.shape(writeData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(writeData[indexX][indexY]))
                    file.write('\n')

            startPosition += numpy.shape(data)[0]
