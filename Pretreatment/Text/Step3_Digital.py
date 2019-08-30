import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Text_Step1_RawText/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Data/Text_Step3_Digital/'
    dictionaryData = numpy.genfromtxt(fname='Dictionary.csv', dtype=str, delimiter=',')

    dictionary = {}
    for sample in dictionaryData:
        dictionary[sample[0]] = int(sample[1])
    print(dictionary)

    for foldname in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            with open(os.path.join(loadpath, foldname, filename), 'r') as file:
                data = file.readlines()
            with open(os.path.join(savepath, foldname, filename), 'w') as file:
                for sample in data:
                    sample = sample.replace('[', '')
                    sample = sample.replace(']', '')
                    sample = sample.replace('_', '')
                    sample = sample.replace('<', '')
                    sample = sample.replace('>', '')

                    sample = sample.replace('  ', ' ')
                    sample = sample.replace('  ', ' ')
                    sample = sample.replace('  ', ' ')
                    sample = sample.replace('  ', ' ')
                    sample = sample.replace('  ', ' ')

                    for words in sample[0:-1].split(' '):
                        if words in dictionary.keys():
                            file.write(str(dictionary[words]) + ',')
                    file.write('\n')
            # print(data)

            # exit()
