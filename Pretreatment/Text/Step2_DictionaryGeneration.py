import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Text_Step1_RawText/'

    dictionary = {}
    counter = 1
    with open('Dictionary.csv', 'w') as writeFile:
        for foldname in os.listdir(loadpath):
            for filename in os.listdir(os.path.join(loadpath, foldname)):
                with open(os.path.join(loadpath, foldname, filename), 'r') as file:
                    data = file.read()
                    data = data.replace('[', '')
                    data = data.replace(']', '')
                    data = data.replace('_', '')
                    data = data.replace('<', '')
                    data = data.replace('>', '')
                    # data = data.replace('\num', ' ')
                    data = data.replace('\n', ' ')

                    data = data.replace('  ', ' ')
                    data = data.replace('  ', ' ')
                    data = data.replace('  ', ' ')
                    data = data.replace('  ', ' ')
                    data = data.replace('  ', ' ')

                    print(foldname, filename)
                    data = data.split(' ')
                    for sample in data:
                        if sample == '': continue
                        if sample in dictionary.keys(): continue
                        dictionary[sample] = counter
                        writeFile.write(sample + ',' + str(counter) + '\n')
                        counter += 1
                    # exit()
