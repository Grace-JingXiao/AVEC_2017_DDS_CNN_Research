import tensorflow


def AttentionTransform_OneDimension_Initializer(dataInput, sourceMap, targetMap, scopeName, blstmFlag=True):
    with tensorflow.variable_scope(scopeName), tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput

        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(tensorflow.shape(networkParameter['DataInput'], name='Shape'))
        ############################################################################

        networkParameter['LossWeight'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.nn.tanh, name='LossWeight') + 1
        networkParameter['AttentionDistance'] = tensorflow.abs(sourceMap - targetMap, name='AttentionDistance')
        networkParameter['LossRaw'] = tensorflow.multiply(
            networkParameter['LossWeight'], networkParameter['AttentionDistance'][:, :, tensorflow.newaxis],
            name='LossRaw')
        if scopeName.find('MA') != -1:
            networkParameter['Loss'] = tensorflow.reduce_mean(
                input_tensor=networkParameter['LossRaw'], axis=1, name='Loss')
        else:
            networkParameter['Loss'] = tensorflow.reduce_sum(
                input_tensor=networkParameter['LossRaw'], axis=1, name='Loss')
    return networkParameter
