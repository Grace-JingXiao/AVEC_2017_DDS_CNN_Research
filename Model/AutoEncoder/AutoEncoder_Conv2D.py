import numpy
import tensorflow
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Single


class AutoEncoder_Conv2D(NeuralNetwork_Base):
    def __init__(self, trainData, batchSize=64, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        super(AutoEncoder_Conv2D, self).__init__(
            trainData=trainData, trainLabel=None, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[None, 1000, 40], name='dataInput')
        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=16, kernel_size=[5, 5], strides=[2, 2],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv')
        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_Conv'], filters=32, kernel_size=[5, 5], strides=[4, 4], padding='SAME',
            activation=tensorflow.nn.relu, name='Layer2nd_Conv')
        self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer2nd_Conv'], filters=128, kernel_size=[5, 5], strides=[5, 5], padding='SAME',
            activation=tensorflow.nn.relu, name='Layer3rd_Conv')
        self.parameters['Layer4th_Transpose'] = tensorflow.layers.conv2d_transpose(
            inputs=self.parameters['Layer3rd_Conv'], filters=32, kernel_size=[5, 5], strides=[5, 5], padding='SAME',
            activation=tensorflow.nn.relu, name='Layer4th_Transpose')
        self.parameters['Layer5th_Transpose'] = tensorflow.layers.conv2d_transpose(
            inputs=self.parameters['Layer4th_Transpose'], filters=16, kernel_size=[5, 5], strides=[4, 4],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer5th_Transpose')
        self.parameters['Predict'] = tensorflow.layers.conv2d_transpose(
            inputs=self.parameters['Layer5th_Transpose'], filters=1, kernel_size=[5, 5], strides=[2, 2], padding='SAME',
            activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
            labels=self.dataInput[:, :, :, tensorflow.newaxis], predictions=self.parameters['Predict'], weights=10)
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        batchData = self.data[0:self.batchSize]
        result = self.session.run(fetches=self.parameters['Loss'], feed_dict={self.dataInput: batchData})
        print(numpy.shape(result))
        print(result)

    def Train(self, logName):
        trainData = Shuffle_Single(self.data)

        with open(logName, 'w') as file:
            startPosition = 0
            totalLoss = 0.0
            while startPosition < numpy.shape(trainData)[0]:
                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train], feed_dict={
                    self.dataInput: trainData[startPosition:startPosition + self.batchSize]})
                startPosition += self.batchSize
                print('\rTraining %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
                totalLoss += loss
                file.write(str(loss) + '\n')
        return totalLoss
