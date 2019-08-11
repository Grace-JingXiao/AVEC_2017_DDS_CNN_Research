import tensorflow


def RNN_StandardAttentionInitializer(
        dataInput, seqInput, scopeName, hiddenNoduleNumber, attentionScope=None, blstmFlag=True):
    with tensorflow.variable_scope(scopeName), tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput
        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(
            tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['DataReshape'] = tensorflow.reshape(
            tensor=networkParameter['DataInput'],
            shape=[networkParameter['BatchSize'] * networkParameter['TimeStep'], hiddenNoduleNumber],
            name='Reshape')
        networkParameter['AttentionWeight'] = tensorflow.layers.dense(
            inputs=networkParameter['DataReshape'], units=1, activation=tensorflow.nn.tanh,
            name='Weight')
        networkParameter['AttentionReshape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight'],
            shape=[networkParameter['BatchSize'], networkParameter['TimeStep']],
            name='WeightReshape')

        if seqInput is not None:
            with tensorflow.name_scope('AttentionMask'):
                networkParameter['AttentionMask'] = (tensorflow.sequence_mask(
                    lengths=seqInput, maxlen=500, dtype=tensorflow.float32) * 2 - tensorflow.ones(
                    shape=[networkParameter['BatchSize'], 500], dtype=tensorflow.float32)) * 9999
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
