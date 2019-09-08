import os
import numpy
import matplotlib.pylab as plt

MAX_SENTENCE = 128
MAX_FRAME = 1000

if __name__ == '__main__':
    partName = 'pose'
    loadpath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step4_%s_Normalization/' % partName
    savepath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Data_%s/' % partName
    os.makedirs(savepath)

    usualShape = 0

    for foldname in os.listdir(loadpath):
        totalData, totalSeq = [], []
        for filename in os.listdir(os.path.join(loadpath, foldname))[0:MAX_SENTENCE]:
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            if usualShape == 0:
                usualShape = numpy.shape(data)[1]
            else:
                data = numpy.reshape(data, [-1, usualShape])

            totalSeq.append(min(MAX_FRAME, numpy.shape(data)[0]))
            if numpy.shape(data)[0] < MAX_FRAME:
                data = numpy.concatenate([data, numpy.zeros([MAX_FRAME - numpy.shape(data)[0], numpy.shape(data)[1]])],
                                         axis=0)
            else:
                data = data[0:MAX_FRAME]

            totalData.append(data)
        print(foldname, numpy.shape(totalData))
        numpy.save(savepath + foldname + '.npy', totalData)
        numpy.save(savepath + foldname + '_Seq.npy', totalSeq)
