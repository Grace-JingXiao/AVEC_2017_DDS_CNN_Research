import tensorflow


def RNN_LocalAttentionInitializer(dataInput, seqInput, scopeName, hiddenNoduleNumber, attentionScope=None,
                                  blstmFlag=True):
    with tensorflow.variable_scope(scopeName), tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput

        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(tensorflow.shape(networkParameter['DataInput'], name='Shape'))
        networkParameter['DataSupplement'] = tensorflow.concat([networkParameter['DataInput'], tensorflow.zeros(
            shape=[networkParameter['BatchSize'], attentionScope * 2 + 1, networkParameter['HiddenNoduleNumber']])],
                                                               axis=1, name='DataSupplement')

        ############################################################################

        networkParameter['DataPile_0'] = tensorflow.concat(
            [networkParameter['DataSupplement'][:, 0:-2 * attentionScope - 1, :],
             networkParameter['DataSupplement'][:, 1:-2 * attentionScope, :]], axis=2, name='DataPile_0')
        for times in range(1, 2 * attentionScope):
            networkParameter['DataPile_%d' % times] = tensorflow.concat(
                [networkParameter['DataPile_%d' % (times - 1)],
                 networkParameter['DataSupplement'][:, times + 1:-2 * attentionScope + times, :]], axis=2,
                name='DataPile_%d' % times)
        networkParameter['DataPileFinal'] = networkParameter['DataPile_%d' % (2 * attentionScope - 1)]

        ############################################################################

        networkParameter['DataReshape'] = tensorflow.reshape(
            tensor=networkParameter['DataPileFinal'],
            shape=[networkParameter['BatchSize'] * networkParameter['TimeStep'],
                   (2 * attentionScope + 1) * networkParameter['HiddenNoduleNumber']], name='DataReshape')
        networkParameter['DataReshape'].set_shape([None, (2 * attentionScope + 1) * hiddenNoduleNumber])
        networkParameter['AttentionWeight'] = tensorflow.layers.dense(
            inputs=networkParameter['DataReshape'], units=1, activation=tensorflow.nn.tanh,
            name='DataWeight')
        networkParameter['AttentionReshape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight'],
            shape=[networkParameter['BatchSize'], networkParameter['TimeStep']],
            name='DataWeightReshape')

        if seqInput is not None:
            with tensorflow.name_scope('AttentionMask'):
                networkParameter['AttentionMask'] = (tensorflow.sequence_mask(
                    lengths=seqInput, maxlen=networkParameter['TimeStep'],
                    dtype=tensorflow.float32) * 2 - tensorflow.ones(
                    shape=[networkParameter['BatchSize'], networkParameter['TimeStep']],
                    dtype=tensorflow.float32)) * 9999
            networkParameter['AttentionReshapeWithMask'] = tensorflow.minimum(
                networkParameter['AttentionReshape'], networkParameter['AttentionMask'],
                name='AttentionReshapeWithMask')
        else:
            networkParameter['AttentionReshapeWithMask'] = networkParameter['AttentionReshape']

        networkParameter['AttentionFinal'] = tensorflow.nn.softmax(
            logits=networkParameter['AttentionReshapeWithMask'], name='AttentionFinal')
        networkParameter['AttentionSupplement'] = tensorflow.tile(
            input=networkParameter['AttentionFinal'][:, :, tensorflow.newaxis],
            multiples=[1, 1, hiddenNoduleNumber],
            name='AttentionSupplement')
        networkParameter['FinalResult_Media'] = tensorflow.multiply(x=networkParameter['DataInput'],
                                                                    y=networkParameter['AttentionSupplement'],
                                                                    name='FinalResult_Media')
        networkParameter['FinalResult'] = tensorflow.reduce_sum(input_tensor=networkParameter['FinalResult_Media'],
                                                                axis=1, name='FinalResult')

    return networkParameter
