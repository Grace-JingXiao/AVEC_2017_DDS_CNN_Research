import numpy
import tensorflow
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Triple
from tensorflow.contrib import rnn


class CNN_CTC(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeq, batchSize=32, learningRate=1E-3, startFlag=True,
                 graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.seq = trainSeq
        self.hiddenNoduleNumber = 128
        self.rnnLayers = 2
        super(CNN_CTC, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1000, 40], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None], name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]
        self.parameters['TargetIndex'] = tensorflow.where(
            tensorflow.not_equal(self.labelInput, 0), name='TargetIndex')
        self.parameters['TargetValue'] = tensorflow.gather_nd(self.labelInput, self.parameters['TargetIndex'])
        self.parameters['TargetSparse'] = tensorflow.SparseTensor(
            indices=self.parameters['TargetIndex'], values=self.parameters['TargetValue'],
            dense_shape=tensorflow.shape(self.labelInput, out_type=tensorflow.int64))

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv')
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.average_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer1st_MaxPooling')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_MaxPooling'], filters=64, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer2nd_Conv')
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.average_pooling2d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer2nd_MaxPooling')
        self.parameters['Layer2nd_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['Layer2nd_MaxPooling'], shape=[-1, 250, 640], name='Layer2nd_Reshape')

        self.parameters['BLSTM_FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNoduleNumber) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNoduleNumber) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_Output'], self.parameters['BLSTM_FinalState'] = \
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=self.parameters['BLSTM_FW_Cell'], cell_bw=self.parameters['BLSTM_BW_Cell'],
                inputs=self.parameters['Layer2nd_Reshape'], dtype=tensorflow.float32)
        self.parameters['Concat'] = tensorflow.concat(
            [self.parameters['BLSTM_Output'][0], self.parameters['BLSTM_Output'][1]], axis=2)

        self.parameters['Logits'] = tensorflow.layers.dense(
            inputs=self.parameters['Concat'], units=41, activation=None)
        self.parameters['Loss'] = tensorflow.nn.ctc_loss(
            labels=self.parameters['TargetSparse'], inputs=self.parameters['Logits'], sequence_length=self.seqInput,
            time_major=False, ignore_longer_outputs_than_inputs=True)
        self.parameters['Cost'] = tensorflow.reduce_mean(self.parameters['Loss'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Cost'])

    def Valid(self):
        batchData = self.data[0:self.batchSize]
        batchSeq = self.seq[0:self.batchSize]
        batchLabel = self.__LabelPretreatment(self.label[0:self.batchSize])

        result = self.session.run(
            fetches=self.parameters['Concat'],
            feed_dict={self.dataInput: batchData, self.labelInput: batchLabel, self.seqInput: batchSeq})
        print(result)
        print(numpy.shape(result))

    def __LabelPretreatment(self, label):
        batchLabel = []
        maxLen = 0
        for sample in label:
            if maxLen < len(sample): maxLen = len(sample)
        for sample in label:
            if len(sample) < maxLen:
                batchLabel.append(numpy.concatenate([sample, numpy.zeros(maxLen - len(sample))]))
            else:
                batchLabel.append(sample)
        return batchLabel

    def Train(self):
        trainData, trainLabel, trainSeq = Shuffle_Triple(self.data, self.label, self.seq)
        startPosition, totalLoss = 0, 0.0
        while startPosition < numpy.shape(self.data)[0]:
            batchData = trainData[startPosition:startPosition + self.batchSize]
            batchLabel = self.__LabelPretreatment(trainLabel[startPosition:startPosition + self.batchSize])
            batchSeq = trainSeq[startPosition:startPosition + self.batchSize]

            loss, _ = self.session.run(
                fetches=[self.parameters['Cost'], self.train],
                feed_dict={self.dataInput: batchData, self.labelInput: batchLabel, self.seqInput: batchSeq})
            print('\rTraining %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
            totalLoss += loss
            # startPosition += self.batchSize
        return totalLoss
