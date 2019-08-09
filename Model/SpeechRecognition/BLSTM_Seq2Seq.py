import numpy
import tensorflow
from Model.Base import NeuralNetwork_Base
from tensorflow.contrib import rnn, seq2seq


class BLSTM_Seq2Seq(NeuralNetwork_Base):
    def __init__(self, trainData, trainSeq, trainLabel, hiddenNoduleNumber, batchSize=32, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.dataSeq = trainSeq
        self.hiddenNoduleNumber = hiddenNoduleNumber
        super(BLSTM_Seq2Seq, self).__init__(
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
            initial_value=tensorflow.truncated_normal([50, 256]), dtype=tensorflow.float32, name='EmbeddingDictionary')
        self.parameters['EmbeddingResult'] = tensorflow.nn.embedding_lookup(
            params=self.parameters['EmbeddingDictionary'], ids=self.labelInput, name='EmbeddingResult')

        with tensorflow.name_scope('Encoder'):
            self.parameters['Encoder_FW_Cell'] = rnn.LSTMCell(num_units=self.hiddenNoduleNumber, name='Encoder_FW_Cell')
            self.parameters['Encoder_BW_Cell'] = rnn.LSTMCell(num_units=self.hiddenNoduleNumber, name='Encoder_BW_Cell')
            [self.parameters['Encoder_FW_Output'], self.parameters['Encoder_BW_Output']], \
            [self.parameters['Encoder_FW_FinalState'], self.parameters['Encoder_BW_FinalState']] = \
                tensorflow.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.parameters['Encoder_FW_Cell'], cell_bw=self.parameters['Encoder_BW_Cell'],
                    inputs=self.dataInput, sequence_length=self.dataSeqInput, dtype=tensorflow.float32)
            self.parameters['EncoderOutput'] = tensorflow.concat(
                [self.parameters['Encoder_FW_Output'], self.parameters['Encoder_BW_Output']], axis=2,
                name='EncoderOutput')
            self.parameters['Encoder_FinalState_C'] = tensorflow.concat(
                [self.parameters['Encoder_FW_FinalState'].c, self.parameters['Encoder_BW_FinalState'].c], axis=1,
                name='Encoder_FinalState_C')
            self.parameters['Encoder_FinalState_H'] = tensorflow.concat(
                [self.parameters['Encoder_FW_FinalState'].h, self.parameters['Encoder_BW_FinalState'].h], axis=1,
                name='Encoder_FinalState_H')
            self.parameters['Encoder_FinalState'] = rnn.LSTMStateTuple(
                c=self.parameters['Encoder_FinalState_C'], h=self.parameters['Encoder_FinalState_H'])

        #################################################################################

        self.parameters['Helper'] = seq2seq.GreedyEmbeddingHelper(
            embedding=self.parameters['EmbeddingDictionary'],
            start_tokens=tensorflow.ones(self.batchSize, dtype=tensorflow.int32) * 40,
            end_token=0)
        self.parameters['Decoder_Cell'] = rnn.LSTMCell(num_units=2 * self.hiddenNoduleNumber)
        self.parameters['Decoder'] = seq2seq.BasicDecoder(
            cell=self.parameters['Decoder_Cell'], helper=self.parameters['Helper'],
            initial_state=self.parameters['Encoder_FinalState'])
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
        batchData = self.data[0:self.batchSize]
        batchDataSeq = self.dataSeq[0:self.batchSize]
        batchLabel, batchLabelSeq = self.__LabelPretreatment(treatLabel=self.label[0:self.batchSize])

        while True:
            result, _ = self.session.run(
                fetches=[self.parameters['Loss'], self.train],
                feed_dict={self.dataInput: batchData, self.dataSeqInput: batchDataSeq, self.labelInput: batchLabel,
                           self.labelSeqInput: batchLabelSeq})
            print('\rLoss = %f' % result, end='')

    def Test(self):
        pass
