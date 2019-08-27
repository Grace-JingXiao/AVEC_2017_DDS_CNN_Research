import numpy
import tensorflow
from tensorflow.contrib import rnn
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Triple
from Model.AttentionTransform.AttentionTransform_OneDimension import AttentionTransform_OneDimension_Initializer

COMPARE = {'RSA': 'AttentionFinal', 'RLA': 'AttentionFinal', 'RMA': 'AttentionWeight_Final'}
POSITION = {'RSA': 18, 'RLA': 18, 'RMA': 24}


class MultiCNNWithSR_NewTransform(NeuralNetwork_Base):
    def __init__(self, trainData, trainSeq, trainLabel, transformWeight, firstAttention, firstAttentionName,
                 firstAttentionScope, secondAttention, secondAttentionName, secondAttentionScope, convSize=2,
                 rnnLayers=1, hiddenNodules=128, batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.dataSeq = trainSeq
        self.transformWeight = transformWeight
        self.convSize = convSize
        self.rnnLayers = rnnLayers
        self.hiddenNodules = hiddenNodules
        self.firstAttention, self.firstAttentionName, self.firstAttentionScope = firstAttention, firstAttentionName, firstAttentionScope
        self.secondAttention, self.secondAttentionName, self.secondAttentionScope = secondAttention, secondAttentionName, secondAttentionScope

        super(MultiCNNWithSR_NewTransform, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1000, 40], name='DataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[1, 1], name='LabelInput')
        self.dataSeqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='SeqInput')

        self.parameters['AttentionResultCurrent_SR'] = []
        for sample in self.convSize:
            self.parameters['Layer1st_Conv_%d_SR' % sample] = tensorflow.layers.conv2d(
                inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[sample, sample],
                strides=[1, 1], padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv_%d_SR' % sample)
            self.parameters['Layer2nd_MaxPooling_%d_SR' % sample] = tensorflow.layers.max_pooling2d(
                inputs=self.parameters['Layer1st_Conv_%d_SR' % sample], pool_size=[3, 3], strides=[2, 2],
                padding='SAME', name='Layer2nd_MaxPooling_%d_SR' % sample)
            self.parameters['Layer3rd_Conv_%d_SR' % sample] = tensorflow.layers.conv2d(
                inputs=self.parameters['Layer2nd_MaxPooling_%d_SR' % sample], filters=16,
                kernel_size=[sample, sample], strides=[1, 1], padding='SAME', activation=tensorflow.nn.relu,
                name='Layer3rd_Conv_%d_SR' % sample)
            self.parameters['Layer4th_Reshape_%d_SR' % sample] = tensorflow.reshape(
                tensor=self.parameters['Layer3rd_Conv_%d_SR' % sample], shape=[-1, 500, 20 * 16],
                name='Layer4th_Reshape_%d_SR' % sample)

            self.parameters['AttentionMechanism_%d_SR' % sample] = self.firstAttention(
                dataInput=self.parameters['Layer4th_Reshape_%d_SR' % sample], seqInput=self.dataSeqInput,
                scopeName=self.firstAttentionName + '_Frame_%d_SR' % sample, hiddenNoduleNumber=16 * 20,
                attentionScope=self.firstAttentionScope, blstmFlag=False)
            self.parameters['AttentionResult_%d_SR' % sample] = \
                self.parameters['AttentionMechanism_%d_SR' % sample][
                    'FinalResult']
            self.parameters['AttentionResultCurrent_SR'].append(self.parameters['AttentionResult_%d_SR' % sample])
        self.parameters['AttentionResultConcat_SR'] = tensorflow.concat(
            self.parameters['AttentionResultCurrent_SR'], axis=1)

        self.parameters['AttentionLoss'] = 0
        self.parameters['AttentionResultCurrent'] = []
        for sample in self.convSize:
            self.parameters['Layer1st_Conv_%d' % sample] = tensorflow.layers.conv2d(
                inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[sample, sample],
                strides=[1, 1], padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv_%d' % sample)
            self.parameters['Layer2nd_MaxPooling_%d' % sample] = tensorflow.layers.max_pooling2d(
                inputs=self.parameters['Layer1st_Conv_%d' % sample], pool_size=[3, 3], strides=[2, 2], padding='SAME',
                name='Layer2nd_MaxPooling_%d' % sample)
            self.parameters['Layer3rd_Conv_%d' % sample] = tensorflow.layers.conv2d(
                inputs=self.parameters['Layer2nd_MaxPooling_%d' % sample], filters=16,
                kernel_size=[sample, sample], strides=[1, 1], padding='SAME', activation=tensorflow.nn.relu,
                name='Layer3rd_Conv_%d' % sample)
            self.parameters['Layer4th_Reshape_%d' % sample] = tensorflow.reshape(
                tensor=self.parameters['Layer3rd_Conv_%d' % sample], shape=[-1, 500, 20 * 16],
                name='Layer4th_Reshape_%d' % sample)
            self.parameters['AttentionMechanism_%d' % sample] = self.firstAttention(
                dataInput=self.parameters['Layer4th_Reshape_%d' % sample], seqInput=self.dataSeqInput,
                scopeName=self.firstAttentionName + '_Frame_%d' % sample, hiddenNoduleNumber=16 * 20,
                attentionScope=self.firstAttentionScope, blstmFlag=False)
            self.parameters['AttentionResult_%d' % sample] = self.parameters['AttentionMechanism_%d' % sample][
                'FinalResult']
            self.parameters['AttentionResultCurrent'].append(self.parameters['AttentionResult_%d' % sample])

            self.parameters['AttentionTransform_%d' % sample] = AttentionTransform_OneDimension_Initializer(
                dataInput=self.parameters['Layer4th_Reshape_%d' % sample],
                sourceMap=self.parameters['AttentionMechanism_%d_SR' % sample][COMPARE[self.firstAttentionName]],
                targetMap=self.parameters['AttentionMechanism_%d' % sample][COMPARE[self.firstAttentionName]],
                scopeName='AttentionTransform_%s_%d' % (self.firstAttentionName, sample), blstmFlag=False)
            self.parameters['AttentionLoss'] = \
                self.parameters['AttentionLoss'] + self.parameters['AttentionTransform_%d' % sample]['Loss']

        self.parameters['AttentionResultConcat'] = tensorflow.concat(
            self.parameters['AttentionResultCurrent'], axis=1, name='AttentionResultConcat')

        self.parameters['BLSTM_FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_Output'], self.parameters['BLSTM_FinalState'] = \
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=self.parameters['BLSTM_FW_Cell'], cell_bw=self.parameters['BLSTM_BW_Cell'],
                inputs=self.parameters['AttentionResultConcat'][tensorflow.newaxis, :, :], dtype=tensorflow.float32)

        self.parameters['BLSTM_AttentionMechanism'] = self.secondAttention(
            dataInput=self.parameters['BLSTM_Output'], seqInput=None, scopeName=self.secondAttentionName + '_Sentence',
            hiddenNoduleNumber=2 * self.hiddenNodules, attentionScope=self.secondAttentionScope, blstmFlag=True)
        self.parameters['BLSTM_Result'] = self.parameters['BLSTM_AttentionMechanism']['FinalResult']
        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['BLSTM_Result'], units=1, activation=None)
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Loss'] + self.transformWeight * self.parameters['AttentionLoss'],
            var_list=tensorflow.global_variables()[POSITION[self.firstAttentionName]:])

    def Load_Part(self, loadpath):
        saver = tensorflow.train.Saver(var_list=tensorflow.global_variables()[0:POSITION[self.firstAttentionName]])
        saver.restore(self.session, loadpath)

    def Valid(self):
        result = self.session.run(
            fetches=self.parameters['Loss'],
            feed_dict={self.dataInput: self.data[0], self.dataSeqInput: self.dataSeq[0],
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
                    feed_dict={self.dataInput: trainData[index], self.dataSeqInput: trainSeq[index],
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
                    feed_dict={self.dataInput: testData[index], self.dataSeqInput: testSeq[index]})
                file.write(str(testLabel[index]) + ',' + str(predict[0][0]) + '\n')
