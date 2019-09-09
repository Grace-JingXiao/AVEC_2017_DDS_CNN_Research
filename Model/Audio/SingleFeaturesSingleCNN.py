import numpy
import tensorflow
from tensorflow.contrib import rnn
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Triple


class SingleCNN_Audio(NeuralNetwork_Base):
    def __init__(self, trainData, trainSeq, trainLabel, firstAttention, firstAttentionName, firstAttentionScope,
                 secondAttention, secondAttentionName, secondAttentionScope, convSize=2, rnnLayers=1, hiddenNodules=128,
                 batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        self.dataSeq = trainSeq
        self.convSize = convSize
        self.rnnLayers = rnnLayers
        self.hiddenNodules = hiddenNodules
        self.firstAttention, self.firstAttentionName, self.firstAttentionScope = firstAttention, firstAttentionName, firstAttentionScope
        self.secondAttention, self.secondAttentionName, self.secondAttentionScope = secondAttention, secondAttentionName, secondAttentionScope

        super(SingleCNN_Audio, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[None, 1000, numpy.shape(self.data[0])[2]], name='DataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[1, 1], name='LabelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='SeqInput')

        self.parameters['Layer0th_Pretreatment'] = tensorflow.layers.dense(
            inputs=self.dataInput, units=128, activation=None, name='Layer0th_Pretreatment')

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer0th_Pretreatment'][:, :, :, tensorflow.newaxis], filters=8,
            kernel_size=[self.convSize, self.convSize], strides=[1, 1], padding='SAME', activation=tensorflow.nn.relu,
            name='Layer1st_Conv')
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=[3, 3], strides=[2, 2], padding='SAME',
            name='Layer2nd_MaxPooling')
        self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer2nd_MaxPooling'], filters=16, kernel_size=[self.convSize, self.convSize],
            strides=[1, 1], padding='SAME', activation=tensorflow.nn.relu, name='Layer3rd_Conv')
        self.parameters['Layer4th_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['Layer3rd_Conv'], shape=[-1, 500, 64 * 16], name='Layer4th_Reshape')

        self.parameters['AttentionMechanism'] = self.firstAttention(
            dataInput=self.parameters['Layer4th_Reshape'], seqInput=self.seqInput,
            scopeName=self.firstAttentionName + '_Frame', hiddenNoduleNumber=64 * 16,
            attentionScope=self.firstAttentionScope, blstmFlag=False)
        self.parameters['AttentionResult'] = self.parameters['AttentionMechanism']['FinalResult']

        self.parameters['BLSTM_FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_Output'], self.parameters['BLSTM_FinalState'] = \
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=self.parameters['BLSTM_FW_Cell'], cell_bw=self.parameters['BLSTM_BW_Cell'],
                inputs=self.parameters['AttentionResult'][tensorflow.newaxis, :, :], dtype=tensorflow.float32)

        self.parameters['BLSTM_AttentionMechanism'] = self.secondAttention(
            dataInput=self.parameters['BLSTM_Output'], seqInput=None, scopeName=self.secondAttentionName + '_Sentence',
            hiddenNoduleNumber=2 * self.hiddenNodules, attentionScope=self.secondAttentionScope, blstmFlag=True)
        self.parameters['BLSTM_Result'] = self.parameters['BLSTM_AttentionMechanism']['FinalResult']
        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['BLSTM_Result'], units=1, activation=None)
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        # print(numpy.shape(self.data[0]))
        result = self.session.run(
            fetches=self.parameters['Loss'],
            feed_dict={self.dataInput: self.data[0], self.seqInput: self.dataSeq[0],
                       self.labelInput: numpy.reshape(self.label[0], [1, 1])})
        print(result)
        print(numpy.shape(result))

    def Train(self, logName):
        trainData, trainLabel, trainSeq = Shuffle_Triple(self.data, self.label, self.dataSeq)

        totalLoss = 0.0
        with open(logName, 'w') as file:
            for index in range(numpy.shape(trainData)[0]):
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss'], self.train],
                    feed_dict={self.dataInput: trainData[index], self.seqInput: trainSeq[index],
                               self.labelInput: numpy.reshape(trainLabel[index], [1, 1])})
                print('\rTrain %d/%d Loss = %f' % (index, numpy.shape(trainData)[0], loss), end='')
                totalLoss += loss
                file.write(str(loss) + '\n')
        return totalLoss

    def Test(self, savepath, testData, testLabel, testSeq):
        with open(savepath, 'w') as file:
            for index in range(numpy.shape(testData)[0]):
                predict = self.session.run(
                    fetches=self.parameters['Predict'],
                    feed_dict={self.dataInput: testData[index], self.seqInput: testSeq[index]})
                file.write(str(testLabel[index]) + ',' + str(predict[0][0]) + '\n')
