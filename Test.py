import numpy
import matplotlib.pylab as plt
import os

if __name__ == '__main__':
    totaldata = []
    loadpath = r'D:\PythonProjects_Data\Exp_Audio\Single_features_RSA_RSA'
    for episode in range(100):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, 'Loss-%04d.csv' % episode), dtype=float, delimiter=',')
        totaldata.append(numpy.average(data))
    plt.plot(totaldata)
    plt.title('Loss Function')
    plt.xlabel('Train Episode')
    plt.ylabel('Huber Loss')
    plt.show()
