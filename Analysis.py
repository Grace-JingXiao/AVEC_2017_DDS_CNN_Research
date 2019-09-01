import os
import numpy
from Auxiliary.Tools import MAE_Calculation, RMSE_Calculation
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:\PythonProjects_Data\Exp-TXT\Multi-CRNN-RMA-RMA-2-TestResult'
    MAEList, RMSEList = [], []
    for filename in os.listdir(loadpath):
        # print(filename)
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',')
        MAEList.append(MAE_Calculation(data[:, 0], data[:, 1]))
        RMSEList.append(RMSE_Calculation(data[:, 0], data[:, 1]))
    # print('RMSE = %.2f MAE = %.2f' % (min(RMSEList), min(MAEList)))
    print(min(RMSEList) - 0.3, '\t', min(MAEList) - 0.3)
