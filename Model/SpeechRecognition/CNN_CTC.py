import numpy
import tensorflow
from Model.Base import NeuralNetwork_Base

CMU_CLASS = 40


class CNN_CTC(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeq, M2M_Attention, M2M_AttentionName, batchSize=32,
                 learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.seq = trainSeq
        self.M2M_Attention, self.M2M_AttentionName = M2M_Attention, M2M_AttentionName
        super(CNN_CTC, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1000, 40], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1], name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv',
            kernel_initializer=tensorflow.random_normal_initializer(mean=0.0, stddev=0.1))
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer1st_MaxPooling')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer2nd_Conv',
            kernel_initializer=tensorflow.random_normal_initializer(mean=0.0, stddev=0.1))
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer2nd_MaxPooling')

        self.parameters['AttentionList_SR'] = self.M2M_Attention(
            dataInput=self.parameters['Layer2nd_MaxPooling'], seqInput=self.seqInput, scopeName=self.M2M_AttentionName)

    def Valid(self):
        result = self.session.run(fetches=self.parameters['AttentionList_SR']['AttentionWeight_EXP '],
                                  feed_dict={self.dataInput: self.data[0:self.batchSize]})
        print(numpy.shape(result))
        with open('log.csv', 'w') as file:
            for indexX in range(250):
                for indexY in range(10):
                    if indexY != 0: file.write(',')
                    file.write(str(result[0][indexX][indexY][0]))
                file.write('\n')
