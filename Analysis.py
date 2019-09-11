import os
import numpy
from Auxiliary.Tools import MAE_Calculation, RMSE_Calculation
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:\PythonProjects_Data\Exp_Audio\Single_features3D_%s_%s_%d-TestResult'
    for part in ['RSA', 'RLA', 'RMA']:
        for scope in [2, 3, 4]:
            MAEList, RMSEList = [], []
            for filename in os.listdir(loadpath % (part, part, scope)):
                # print(filename)
                data = numpy.genfromtxt(fname=os.path.join(loadpath % (part, part, scope), filename), dtype=float,
                                        delimiter=',')
                MAEList.append(MAE_Calculation(data[:, 0], data[:, 1]))
                RMSEList.append(RMSE_Calculation(data[:, 0], data[:, 1]))
            # print('RMSE = %.2f MAE = %.2f' % (min(RMSEList), min(MAEList)))
            print(min(RMSEList), '\t', min(MAEList))
            # print(numpy.argmax(RMSEList), numpy.argmax(MAEList))
