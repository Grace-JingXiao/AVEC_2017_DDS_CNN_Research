import numpy


def MAE_Calculation(label, predict):
    counter = 0
    for index in range(len(label)):
        counter += numpy.abs(label[index] - predict[index])
    return counter / len(label)


def RMSE_Calculation(label, predict):
    counter = 0
    for index in range(len(label)):
        counter += (label[index] - predict[index]) * (label[index] - predict[index])
    return numpy.sqrt(counter / len(label))
