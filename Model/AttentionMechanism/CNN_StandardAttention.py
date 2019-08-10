import tensorflow


def CNN_StandardAttention_Initializer(inputData, inputSeq, attentionScope, hiddenNoduleNumber, scopeName='CSA'):
    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        networkParameter['DataInput'], networkParameter['SeqInput'] = inputData, inputSeq
        networkParameter['BatchSize'], networkParameter['XScope'], networkParameter['YScope'], \
        networkParameter['HiddenNoduleNumber'] = tensorflow.unstack(
            tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['AttentionWeight_Flat'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.nn.tanh, name='AttentionWeight_Flat')

        networkParameter['AttentionMask'] = \
            (tensorflow.sequence_mask(lengths=networkParameter['SeqInput'], maxlen=250, dtype=tensorflow.float32) * 2 - \
             tensorflow.ones(shape=[networkParameter['BatchSize'], 250], dtype=tensorflow.float32)) * 9999
        networkParameter['AttentionMask_Tile'] = tensorflow.tile(
            input=networkParameter['AttentionMask'][:, :, tensorflow.newaxis, tensorflow.newaxis],
            multiples=[1, 1, 40, 1], name='AttentionMask_Tile')
        networkParameter['AttentionMaskTreatResult'] = tensorflow.minimum(
            networkParameter['AttentionWeight_Flat'], networkParameter['AttentionMask_Tile'],
            name='AttentionMaskTreatResult')

        networkParameter['AttentionWeight_Reshape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionMaskTreatResult'], shape=[networkParameter['BatchSize'], -1],
            name='AttentionWeight_Reshape')
        networkParameter['AttentionWeight_SoftMax'] = tensorflow.nn.softmax(
            networkParameter['AttentionWeight_Reshape'], name='AttentionWeight_SoftMax')
        networkParameter['AttentionWeight_Tile'] = tensorflow.tile(
            input=networkParameter['AttentionWeight_SoftMax'][:, :, tensorflow.newaxis],
            multiples=[1, 1, networkParameter['HiddenNoduleNumber']], name='AttentionWeight_Tile')

        networkParameter['DataReshape'] = tensorflow.reshape(
            tensor=networkParameter['DataInput'],
            shape=[networkParameter['BatchSize'], networkParameter['XScope'] * networkParameter['YScope'],
                   networkParameter['HiddenNoduleNumber']], name='DataReshape')

        networkParameter['Data_AddWeight_Raw'] = tensorflow.multiply(
            networkParameter['AttentionWeight_Tile'], networkParameter['DataReshape'], name='Data_AddWeight_Raw')
        networkParameter['FinalResult'] = tensorflow.reduce_sum(
            input_tensor=networkParameter['Data_AddWeight_Raw'], axis=1, name='FinalResult')
        networkParameter['FinalResult'].set_shape([None, hiddenNoduleNumber])
    return networkParameter
