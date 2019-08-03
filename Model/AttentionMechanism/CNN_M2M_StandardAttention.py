import tensorflow


def CNN_M2M_StandardAttention_Initializer(dataInput, seqInput, scopeName):
    with tensorflow.name_scope(scopeName):
        networkParameter = {}
        networkParameter['DataInput'] = dataInput
        networkParameter['SeqInput'] = seqInput

        networkParameter['BatchSize'], networkParameter['XScope'], networkParameter['YScope'], networkParameter[
            'HiddenNoduleNumbers'] = tensorflow.unstack(tensorflow.shape(networkParameter['DataInput']), name='Shape')
        networkParameter['AttentionWeight'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=None, name='AttentionWeight')

        networkParameter['AttentionWeight_EXP'] = tensorflow.exp(networkParameter['AttentionWeight'],
                                                                 name='AttentionWeight_EXP')

    return networkParameter
