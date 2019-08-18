import numpy
import tensorflow
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Single


class AutoEncoder_Conv3D(NeuralNetwork_Base):
    def __init__(self, trainData, batchSize=64, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        super(AutoEncoder_Conv3D, self).__init__(
            trainData=trainData, trainLabel=None, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[128, 1000, 40], name='dataInput')
        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv3d(
            inputs=self.dataInput[tensorflow.newaxis, :, :, :, tensorflow.newaxis], filters=8, kernel_size=[5, 5, 5],
            strides=[2, 2, 2], padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv')
        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv3d(
            inputs=self.parameters['Layer1st_Conv'], filters=32, kernel_size=[5, 5, 5], strides=[4, 4, 4],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer2nd_Conv')
        self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv3d(
            inputs=self.parameters['Layer2nd_Conv'], filters=128, kernel_size=[4, 5, 5], strides=[4, 5, 5],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer3rd_Conv')
        self.parameters['Layer4th_Transpose'] = tensorflow.layers.conv3d_transpose(
            inputs=self.parameters['Layer3rd_Conv'], filters=32, kernel_size=[4, 5, 5], strides=[4, 5, 5],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer4th_Transpose')
        self.parameters['Layer5th_Transpose'] = tensorflow.layers.conv3d_transpose(
            inputs=self.parameters['Layer4th_Transpose'], filters=8, kernel_size=[5, 5, 5], strides=[4, 4, 4],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer5th_Transpose')
        self.parameters['Layer6th_Transpose'] = tensorflow.layers.conv3d_transpose(
            inputs=self.parameters['Layer5th_Transpose'], filters=1, kernel_size=[5, 5, 5], strides=[2, 2, 2],
            padding='SAME', activation=None, name='Layer6th_Transpose')
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
            labels=self.dataInput[tensorflow.newaxis, :, :, :, tensorflow.newaxis],
            predictions=self.parameters['Layer6th_Transpose'], weights=10)
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        # print(numpy.shape(self.data[1])[0])
        result = self.session.run(fetches=self.parameters['Loss'],
                                  feed_dict={self.dataInput: self.data[1]})
        print(numpy.shape(result))
        print(result)

    def Train(self, logName):
        trainData = Shuffle_Single(self.data)
        totalLoss = 0.0

        with open(logName, 'w') as file:
            for index in range(numpy.shape(trainData)[0]):
                sampleData = trainData[index]
                if numpy.shape(sampleData)[0] < 128:
                    sampleData = numpy.concatenate([sampleData, numpy.zeros(
                        [128 - numpy.shape(sampleData)[0], numpy.shape(sampleData)[1], numpy.shape(sampleData)[2]])])
                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                           feed_dict={self.dataInput: sampleData})
                totalLoss += loss
                print('\rTraining %d/%d Loss = %f' % (index, numpy.shape(trainData)[0], loss), end='')
                file.write(str(loss) + '\n')
        return totalLoss

    def MiddleResultGenerate(self, savepath, testData):
        with open(savepath, 'w') as file:
            for index in range(numpy.shape(testData)[0]):
                sampleData = testData[index]
                if numpy.shape(sampleData)[0] < 128:
                    sampleData = numpy.concatenate([sampleData, numpy.zeros(
                        [128 - numpy.shape(sampleData)[0], numpy.shape(sampleData)[1], numpy.shape(sampleData)[2]])])
                result = self.session.run(fetches=self.parameters['Layer3rd_Conv'],
                                          feed_dict={self.dataInput: sampleData})
                result = numpy.reshape(result, [1, 128 * 4 * 25])

                for indexX in range(numpy.shape(result)[1]):
                    if indexX != 0: file.write(',')
                    file.write(str(result[0][indexX]))
                file.write('\n')
