import numpy
import tensorflow
from tensorflow.contrib import rnn
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Double


class SingleLayer(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, attention, attentionName, attentionScope, convSize=2,
                 rnnLayers=1, hiddenNodules=128, batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.convSize = convSize
        self.rnnLayers = rnnLayers
        self.hiddenNodules = hiddenNodules
        self.attention, self.attentionName, self.attentionScope = attention, attentionName, attentionScope

        super(SingleLayer, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[1, 1], name='LabelInput')

        self.parameters['EmbeddingVocabulary'] = tensorflow.Variable(
            initial_value=tensorflow.truncated_normal(shape=[9000, 256]), name='EmbeddingVocabulary')
        self.parameters['EmbeddingData'] = tensorflow.nn.embedding_lookup(
            params=self.parameters['EmbeddingVocabulary'], ids=self.dataInput, name='EmbeddingData')

        self.parameters['AttentionResultCurrent'] = []
        for sample in self.convSize:
            self.parameters['Layer1st_Conv_%d' % sample] = tensorflow.layers.conv2d(
                inputs=self.parameters['EmbeddingData'][tensorflow.newaxis, :, :, tensorflow.newaxis], filters=8,
                kernel_size=[sample, sample], strides=[1, 1], padding='SAME', activation=tensorflow.nn.relu,
                name='Layer1st_Conv_%d' % sample)
            self.parameters['Layer2nd_Conv_%d' % sample] = tensorflow.layers.conv2d(
                inputs=self.parameters['Layer1st_Conv_%d' % sample], filters=16,
                kernel_size=[sample, sample], strides=[1, 1], padding='SAME', activation=tensorflow.nn.relu,
                name='Layer2nd_Conv_%d' % sample)
            self.parameters['Layer3rd_Reshape_%d' % sample] = tensorflow.reshape(
                tensor=self.parameters['Layer2nd_Conv_%d' % sample], shape=[1, -1, 256 * 16],
                name='Layer3rd_Reshape_%d' % sample)

            self.parameters['AttentionMechanism_%d' % sample] = self.attention(
                dataInput=self.parameters['Layer3rd_Reshape_%d' % sample], seqInput=None,
                scopeName=self.attentionName + '_Frame_%d' % sample, hiddenNoduleNumber=256 * 16,
                attentionScope=self.attentionScope, blstmFlag=False)
            self.parameters['AttentionResult_%d' % sample] = self.parameters['AttentionMechanism_%d' % sample][
                'FinalResult']
            self.parameters['AttentionResultCurrent'].append(self.parameters['AttentionResult_%d' % sample])

        self.parameters['AttentionReshape'] = tensorflow.reshape(
            tensor=self.parameters['AttentionResultCurrent'], shape=[1, 4096 * len(self.convSize)],
            name='AttentionReshape')

        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionReshape'], units=1, activation=None)
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        result = self.session.run(fetches=self.parameters['Predict'],
                                  feed_dict={self.dataInput: self.data[0]})
        print(numpy.shape(result))

    def Train(self, logName):
        trainData, trainLabel = Shuffle_Double(self.data, self.label)

        totalLoss = 0.0
        with open(logName, 'w') as file:
            for index in range(numpy.shape(trainData)[0]):
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss'], self.train],
                    feed_dict={self.dataInput: trainData[index],
                               self.labelInput: numpy.reshape(trainLabel[index], [1, 1])})
                print('\rTrain %d/%d Loss = %f' % (index, numpy.shape(trainData)[0], loss), end='')
                totalLoss += loss
                file.write(str(loss) + '\n')
        return totalLoss

    def Test(self, savepath, testData, testLabel):
        with open(savepath, 'w') as file:
            for index in range(numpy.shape(testData)[0]):
                predict = self.session.run(
                    fetches=self.parameters['Predict'],
                    feed_dict={self.dataInput: testData[index]})
                file.write(str(testLabel[index]) + ',' + str(predict[0][0]) + '\n')
