import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Text_Step3_Digital/'

    totalCounter = []
    counter = 0
    for foldname in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            with open(os.path.join(loadpath, foldname, filename), 'r') as file:
                data = file.readlines()
                for index in range(numpy.shape(data)[0]):
                    totalCounter.append(len(data[index].split(',')) - 1)
                    if len(data[index].split(',')) - 1 > 25: counter += 1
            # print(data)
            # exit()
    print(totalCounter)
    print(len(totalCounter))
    print(numpy.average(totalCounter), numpy.median(totalCounter))
    print(counter)
