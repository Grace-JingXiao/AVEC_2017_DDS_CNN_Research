import numpy
import tensorflow
from tensorflow.contrib import rnn, seq2seq
from Auxiliary.Shuffle import Shuffle_Triple
from Model.Base import NeuralNetwork_Base
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttention_Initializer


class CNN_Seq2Seq(NeuralNetwork_Base):
    def __init__(self, trainData, trainSeq, trainLabel, hiddenNoduleNumbers=128, batchSize=32, learningRate=1E-3,
                 startFlag=True,
                 graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.dataSeq = trainSeq
        self.hiddenNoduleNumbers = hiddenNoduleNumbers
        super(CNN_Seq2Seq, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[self.batchSize, 1000, 40], name='dataInput')
        self.dataSeqInput = tensorflow.placeholder(
            dtype=tensorflow.int32, shape=[self.batchSize], name='dataSeqInput')
        self.labelInput = tensorflow.placeholder(
            dtype=tensorflow.int32, shape=[self.batchSize, None], name='labelInput')
        self.labelSeqInput = tensorflow.placeholder(
            dtype=tensorflow.int32, shape=[self.batchSize], name='labelSeqInput')

        self.parameters['EmbeddingDictionary'] = tensorflow.Variable(
            initial_value=tensorflow.truncated_normal([50, 2 * self.hiddenNoduleNumbers]), dtype=tensorflow.float32,
            name='EmbeddingDictionary')

        with tensorflow.name_scope('Encoder'):
            self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
                inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[3, 3], strides=[1, 1],
                padding='SAME', name='Layer1st_Conv')
            self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling2d(
                inputs=self.parameters['Layer1st_Conv'], pool_size=[3, 3], strides=[2, 1], padding='SAME',
                name='Layer1st_MaxPooling')
            self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
                inputs=self.parameters['Layer1st_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
                padding='SAME', name='Layer2nd_Conv')
            self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
                inputs=self.parameters['Layer2nd_Conv'], pool_size=[3, 3], strides=[2, 1], padding='SAME',
                name='Layer2nd_MaxPooling')
            self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv2d(
                inputs=self.parameters['Layer2nd_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
                padding='SAME', name='Layer3rd_Conv')

        ###############################################################################

        self.parameters['AttentionList'] = CNN_StandardAttention_Initializer(
            inputData=self.parameters['Layer3rd_Conv'], inputSeq=self.dataSeqInput, attentionScope=None,
            hiddenNoduleNumber=16, scopeName='CSA')
        self.parameters['AttentionResult'] = self.parameters['AttentionList']['FinalResult']

        ###############################################################################

        self.parameters['DecoderInitialState_C'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResult'], units=2 * self.hiddenNoduleNumbers, activation=None,
            name='DecoderInitialState_C')
        self.parameters['DecoderInitialState_H'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResult'], units=2 * self.hiddenNoduleNumbers, activation=None,
            name='DecoderInitialState_H')
        self.parameters['DecoderInitialState'] = rnn.LSTMStateTuple(
            c=self.parameters['DecoderInitialState_C'], h=self.parameters['DecoderInitialState_H'])

        ###############################################################################

        self.parameters['Helper'] = seq2seq.GreedyEmbeddingHelper(
            embedding=self.parameters['EmbeddingDictionary'],
            start_tokens=tensorflow.ones(self.batchSize, dtype=tensorflow.int32) * 40,
            end_token=0)
        self.parameters['Decoder_Cell'] = rnn.LSTMCell(num_units=2 * self.hiddenNoduleNumbers)
        self.parameters['Decoder'] = seq2seq.BasicDecoder(
            cell=self.parameters['Decoder_Cell'], helper=self.parameters['Helper'],
            initial_state=self.parameters['DecoderInitialState'])

        self.parameters['DecoderOutput'], self.parameters['DecoderFinalState'], self.parameters['DecoderSeqLen'] = \
            seq2seq.dynamic_decode(decoder=self.parameters['Decoder'], output_time_major=False,
                                   maximum_iterations=tensorflow.reduce_max(self.labelSeqInput))

        self.parameters['Logits'] = tensorflow.layers.dense(
            inputs=self.parameters['DecoderOutput'][0], units=50, activation=None, name='Logits')
        # self.parameters['Mask'] = tensorflow.to_float(tensorflow.not_equal(self.labelInput, 0))
        self.parameters['Loss'] = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(
            labels=tensorflow.one_hot(self.labelInput, depth=50, dtype=tensorflow.float32),
            logits=self.parameters['Logits']), name='Loss')
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        batchData = self.data[0:self.batchSize]
        batchDataSeq = self.dataSeq[0:self.batchSize]
        batchLabel, batchLabelSeq = self.__LabelPretreatment(treatLabel=self.label[0:self.batchSize])

        result = self.session.run(
            fetches=self.parameters['Loss'],
            feed_dict={self.dataInput: batchData, self.dataSeqInput: batchDataSeq, self.labelInput: batchLabel,
                       self.labelSeqInput: batchLabelSeq})
        print(result)
        print(numpy.shape(result))

    def __LabelPretreatment(self, treatLabel):
        batchLabel, batchLabelSeq = [], []
        maxLen = 0

        for sample in treatLabel:
            if maxLen < len(sample): maxLen = len(sample)
        for sample in treatLabel:
            batchLabelSeq.append(len(sample))
            if len(sample) < maxLen:
                batchLabel.append(numpy.concatenate([sample, numpy.zeros(maxLen - len(sample))]))
            else:
                batchLabel.append(sample)
        return batchLabel, batchLabelSeq

    def Train(self, logName):
        trainData, trainLabel, trainSeq = Shuffle_Triple(self.data, self.label, self.dataSeq)

        startPosition, totalLoss = 0, 0.0
        with open(logName, 'w') as file:
            while startPosition + self.batchSize < numpy.shape(trainData)[0]:
                batchData = self.data[startPosition:startPosition + self.batchSize]
                batchDataSeq = self.dataSeq[startPosition:startPosition + self.batchSize]
                batchLabel, batchLabelSeq = self.__LabelPretreatment(
                    treatLabel=self.label[startPosition:startPosition + self.batchSize])
                # print(numpy.shape(batchData),numpy.shape(batchDataSeq),numpy.shape(batchLabel),numpy.shape(batchLabelSeq))
                # exit()
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss'], self.train],
                    feed_dict={self.dataInput: batchData, self.dataSeqInput: batchDataSeq, self.labelInput: batchLabel,
                               self.labelSeqInput: batchLabelSeq})
                print('\rTrain %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
                startPosition += self.batchSize
                totalLoss += loss
                file.write(str(loss) + '\n')
        return totalLoss
