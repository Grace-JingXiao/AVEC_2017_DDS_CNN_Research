import numpy
import tensorflow
from Model.Base import NeuralNetwork_Base
from tensorflow.contrib import rnn, seq2seq
from Auxiliary.Shuffle import Shuffle_Triple


class MultiCNN_Seq2Seq(NeuralNetwork_Base):
    def __init__(self, trainData, trainSeq, trainLabel, attention, attentionName, attentionScope, convSize=2,
                 batchSize=32, hiddenNoduleNumbers=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.dataSeq = trainSeq
        self.convSize = convSize
        self.hiddenNoduleNumbers = hiddenNoduleNumbers
        self.attention, self.attentionName, self.attentionScope = attention, attentionName, attentionScope

        super(MultiCNN_Seq2Seq, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1000, 40], name='DataInput')
        self.labelInput_SR = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None], name='LabelInput_SR')
        self.dataSeqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='SeqInput')
        self.labelSeqInput_SR = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='LabelSeqInput')

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

            self.parameters['AttentionMechanism_%d_SR' % sample] = self.attention(
                dataInput=self.parameters['Layer4th_Reshape_%d_SR' % sample], seqInput=self.dataSeqInput,
                scopeName=self.attentionName + '_Frame_%d_SR' % sample, hiddenNoduleNumber=16 * 20,
                attentionScope=self.attentionScope, blstmFlag=False)
            self.parameters['AttentionResult_%d_SR' % sample] = self.parameters['AttentionMechanism_%d_SR' % sample][
                'FinalResult']
            self.parameters['AttentionResultCurrent_SR'].append(self.parameters['AttentionResult_%d_SR' % sample])

        self.parameters['AttentionResultConcat_SR'] = tensorflow.concat(
            self.parameters['AttentionResultCurrent_SR'], axis=1)

        #####################################################################################
        # Encoder Completed
        #####################################################################################

        self.parameters['DecoderInitialState_C_SR'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResultConcat_SR'], units=2 * self.hiddenNoduleNumbers, activation=None,
            name='DecoderInitialState_C_SR')
        self.parameters['DecoderInitialState_H_SR'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResultConcat_SR'], units=2 * self.hiddenNoduleNumbers, activation=None,
            name='DecoderInitialState_H_SR')
        self.parameters['DecoderInitialState_SR'] = rnn.LSTMStateTuple(
            c=self.parameters['DecoderInitialState_C_SR'], h=self.parameters['DecoderInitialState_H_SR'])

        self.parameters['EmbeddingDictionary_SR'] = tensorflow.Variable(
            initial_value=tensorflow.truncated_normal([50, 2 * self.hiddenNoduleNumbers]), dtype=tensorflow.float32,
            name='EmbeddingDictionary_SR')

        self.parameters['Helper_SR'] = seq2seq.GreedyEmbeddingHelper(
            embedding=self.parameters['EmbeddingDictionary_SR'],
            start_tokens=tensorflow.ones(self.batchSize, dtype=tensorflow.int32) * 40,
            end_token=0)
        self.parameters['Decoder_Cell_SR'] = rnn.LSTMCell(num_units=2 * self.hiddenNoduleNumbers)
        self.parameters['Decoder_SR'] = seq2seq.BasicDecoder(
            cell=self.parameters['Decoder_Cell_SR'], helper=self.parameters['Helper_SR'],
            initial_state=self.parameters['DecoderInitialState_SR'])

        self.parameters['DecoderOutput_SR'], self.parameters['DecoderFinalState_SR'], self.parameters[
            'DecoderSeqLen_SR'] = seq2seq.dynamic_decode(
            decoder=self.parameters['Decoder_SR'], output_time_major=False,
            maximum_iterations=tensorflow.reduce_max(self.labelSeqInput_SR))
        self.parameters['Logits_SR'] = tensorflow.layers.dense(
            inputs=self.parameters['DecoderOutput_SR'][0], units=50, activation=None, name='Logits_SR')
        # self.parameters['Mask'] = tensorflow.to_float(tensorflow.not_equal(self.labelInput, 0))
        self.parameters['Loss_SR'] = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(
            labels=tensorflow.one_hot(self.labelInput_SR, depth=50, dtype=tensorflow.float32),
            logits=self.parameters['Logits_SR']), name='Loss_SR')
        self.train_SR = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss_SR'])

    def Valid(self):
        batchData = self.data[0:self.batchSize]
        batchDataSeq = self.dataSeq[0:self.batchSize]
        batchLabel, batchLabelSeq = self.__LabelPretreatment(treatLabel=self.label[0:self.batchSize])

        result = self.session.run(
            fetches=self.parameters['Loss_SR'],
            feed_dict={self.dataInput: batchData, self.dataSeqInput: batchDataSeq, self.labelInput_SR: batchLabel,
                       self.labelSeqInput_SR: batchLabelSeq})
        print(result)
        # print(numpy.shape(result))

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
                    fetches=[self.parameters['Loss_SR'], self.train_SR],
                    feed_dict={self.dataInput: batchData, self.dataSeqInput: batchDataSeq,
                               self.labelInput_SR: batchLabel, self.labelSeqInput_SR: batchLabelSeq})
                print('\rTrain %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
                startPosition += self.batchSize
                totalLoss += loss
                file.write(str(loss) + '\n')
        return totalLoss

    def MiddleResultGeneration(self, testData, testLabel, testSeq):
        startPosition, totalLoss = 0, 0.0
        while startPosition + self.batchSize < numpy.shape(testData)[0]:
            batchData = testData[startPosition:startPosition + self.batchSize]
            batchDataSeq = testSeq[startPosition:startPosition + self.batchSize]
            batchLabel, batchLabelSeq = self.__LabelPretreatment(
                treatLabel=testLabel[startPosition:startPosition + self.batchSize])
            result2, result3, result4 = self.session.run(
                fetches=[self.parameters['AttentionMechanism_2_SR']['AttentionFinal'],
                         self.parameters['AttentionMechanism_3_SR']['AttentionFinal'],
                         self.parameters['AttentionMechanism_4_SR']['AttentionFinal']],
                feed_dict={self.dataInput: batchData, self.dataSeqInput: batchDataSeq,
                           self.labelInput_SR: batchLabel, self.labelSeqInput_SR: batchLabelSeq})
            import matplotlib.pylab as plt
            plt.subplot(111)
            plt.title('Speech Recognition Attention Map Visualization')

            plt.subplot(131)
            plt.imshow(result2[0:64, 0:50])
            plt.xlabel('Conv 2X2')
            plt.ylabel('Sentence')
            plt.subplot(132)
            plt.imshow(result3[0:64, 0:50])
            plt.xlabel('Frames')
            plt.ylabel('Conv 3X3')
            plt.subplot(133)
            plt.imshow(result4[0:64, 0:50])
            plt.xlabel('Frames')
            plt.ylabel('Conv 4X4')
            # plt.colorbar()
            plt.show()
            exit()
