import os
import numpy

if __name__ == '__main__':
    data = numpy.genfromtxt(
        r'D:\PythonProjects_Data\AVEC2017-OtherFeatures\Step4_AUs_Normalization\300_P\Participant_0007.csv',
        dtype=float, delimiter=',')
    print(numpy.shape(data))
