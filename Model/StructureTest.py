import tensorflow
import numpy
from Model.Base import NeuralNetwork_Base


class StructureTest(NeuralNetwork_Base):
    def __init__(self):
        super(StructureTest, self).__init__(trainData=None, trainLabel=None)

    def BuildNetwork(self, learningRate):
        self.predictInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, 4])
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None])
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None])

        self.variable = tensorflow.Variable(initial_value=tensorflow.truncated_normal(shape=[2, 10, 4]))

        self.parameters['TargetIndex'] = tensorflow.where(
            tensorflow.not_equal(self.labelInput, 0), name='TargetIndex')
        self.parameters['TargetValue'] = tensorflow.gather_nd(self.labelInput, self.parameters['TargetIndex'])
        self.parameters['TargetSparse'] = tensorflow.SparseTensor(
            indices=self.parameters['TargetIndex'], values=self.parameters['TargetValue'],
            dense_shape=tensorflow.shape(self.labelInput, out_type=tensorflow.int64))
        self.parameters['Loss'] = tensorflow.nn.ctc_loss(
            labels=self.parameters['TargetSparse'], inputs=self.variable, sequence_length=self.seqInput,
            time_major=False)
        self.parameters['Cost'] = tensorflow.reduce_mean(self.parameters['Loss'])
        self.train = tensorflow.train.AdamOptimizer().minimize(self.parameters['Cost'])

    def Valid(self):
        label = [[1, 2], [2, 1]]
        predict = [[[-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, 1, -1]], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]]
        print(numpy.shape(predict), numpy.shape(label))

        for episode in range(10000):
            loss, _ = self.session.run(fetches=[self.parameters['Cost'], self.train],
                                       feed_dict={self.labelInput: label, self.seqInput: [8, 10]})
            print(episode, loss)
        result = self.session.run(fetches=self.variable)
        print(result)


if __name__ == '__main__':
    classifier = StructureTest()
    classifier.Valid()
