import os
import numpy

MAX_COUNTER = 25

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Text_Step3_Digital/'
    savepath = 'D:/PythonProjects_Data/Data_Text/'

    for foldname in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            with open(os.path.join(loadpath, foldname, filename), 'r') as file:
                data = file.readlines()
            with open(os.path.join(savepath, foldname, filename), 'w') as file:
                with open(os.path.join(savepath, foldname, filename.replace('.csv', '_Seq.csv')), 'w') as seqFile:
                    for sample in data:
                        treatData = sample[0:-2].split(',')
                        if len(treatData) >= MAX_COUNTER:
                            treatData = treatData[0:MAX_COUNTER]
                            seqFile.write(str(MAX_COUNTER) + '\n')
                        else:
                            seqFile.write(str(len(treatData)) + '\n')
                            while len(treatData) < MAX_COUNTER:
                                treatData.append('0')
                        # print(treatData)

                        for index in range(len(treatData)):
                            if index != 0: file.write(',')
                            file.write(treatData[index])
                        file.write('\n')
                # exit()
