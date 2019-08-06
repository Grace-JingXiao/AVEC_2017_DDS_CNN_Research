import tensorflow
import numpy
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Triple


class CTC(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, rnnLayers=1,
                 batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        self.featureShape = featureShape
        self.seqLength = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        self.rnnLayer = rnnLayers
        super(CTC, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                  learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                  graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This is the simplest test of CTC Model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def __LabelPretreatment(self, treatLabel):
        result, maxLen = [], 0
        for sample in treatLabel:
            if maxLen < len(sample): maxLen = len(sample)
        for sample in treatLabel:
            if len(sample) < maxLen:
                concatList = sample.copy()
                concatList.extend(numpy.zeros(maxLen - len(concatList), dtype=int))
                result.append(concatList)
            else:
                result.append(sample)
        # print(numpy.shape(result))
        return result

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='Inputs')
        self.targetsInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None], name='Targets')
        self.seqLengthInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=None, name='seqLength')

        self.parameters['Shape'] = tensorflow.shape(self.dataInput, name='Shape', out_type=tensorflow.int32)
        self.parameters['BatchSize'], self.parameters['TimeStep'] = \
            self.parameters['Shape'][0], self.parameters['Shape'][1]
        self.parameters['TargetIndex'] = tensorflow.where(
            tensorflow.not_equal(self.targetsInput, 0), name='TargetIndex')
        self.parameters['TargetValue'] = tensorflow.gather_nd(self.targetsInput, self.parameters['TargetIndex'])
        self.parameters['TargetSparse'] = tensorflow.SparseTensor(
            indices=self.parameters['TargetIndex'], values=self.parameters['TargetValue'],
            dense_shape=tensorflow.shape(self.targetsInput, out_type=tensorflow.int64))

        ###################################################################################################
        # RNN Start
        ###################################################################################################

        self.rnnCell = []
        for layers in range(self.rnnLayer):
            self.parameters['RNN_Cell_Layer' + str(layers)] = tensorflow.contrib.rnn.LSTMCell(self.hiddenNodules)
            self.rnnCell.append(self.parameters['RNN_Cell_Layer' + str(layers)])
        self.parameters['Stack'] = tensorflow.contrib.rnn.MultiRNNCell(self.rnnCell)

        self.parameters['RNN_Outputs'], _ = tensorflow.nn.dynamic_rnn(
            cell=self.parameters['Stack'], inputs=self.dataInput, sequence_length=self.seqLengthInput,
            dtype=tensorflow.float32)

        ###################################################################################################
        # CTC Start
        ###################################################################################################

        self.parameters['Logits'] = tensorflow.layers.dense(
            inputs=self.parameters['RNN_Outputs'], activation=tensorflow.nn.softmax, units=self.numClass, name='logits')

        self.parameters['Loss'] = tensorflow.nn.ctc_loss(
            labels=self.parameters['TargetSparse'], inputs=self.parameters['Logits'],
            sequence_length=self.seqLengthInput, time_major=False)

        self.parameters['Cost'] = tensorflow.reduce_mean(self.parameters['Loss'], name='Cost')
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate). \
            minimize(self.parameters['Loss'])

        # self.parameters['Decode'], self.parameters['Log_Prob'] = tensorflow.nn.ctc_greedy_decoder(
        #     inputs=self.parameters['Logits_TimeMajor'], sequence_length=self.seqLengthInput)

    def Valid(self):
        trainData, trainLabel, trainSeq = self.data, self.label, self.seqLength
        batchData = trainData[0:self.batchSize]
        batchLabel = self.__LabelPretreatment(trainLabel[0:self.batchSize])
        batchSeq = trainSeq[0:self.batchSize]

        result = self.session.run(
            fetches=self.seqLengthInput,
            feed_dict={self.dataInput: batchData, self.seqLengthInput: batchSeq, self.targetsInput: batchLabel})
        print(result)
        print(numpy.shape(result))
        print('Done')

    def Train(self, logName):
        trainData, trainLabel, trainSeq = Shuffle_Triple(self.data, self.label, self.seqLength)

        startPosition, totalLoss = 0, 0.0
        while startPosition < numpy.shape(trainData)[0]:
            batchData, batchLabel, batchSeq = \
                trainData[startPosition:startPosition + self.batchSize], \
                self.__LabelPretreatment(trainLabel[startPosition:startPosition + self.batchSize]), \
                trainSeq[startPosition:startPosition + self.batchSize]

            loss, _ = self.session.run(
                fetches=[self.parameters['Cost'], self.train],
                feed_dict={self.dataInput: batchData, self.targetsInput: batchLabel, self.seqLengthInput: batchSeq})
            print('\rTraining %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
            startPosition += self.batchSize
            totalLoss += loss
        return totalLoss
