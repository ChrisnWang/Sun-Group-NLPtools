from keras.layers import Dense, Lambda, dot, Activation, concatenate
from keras.layers import Layer
from keras import Model
from keras.layers import Dropout, Embedding, LSTM


class Attention(Layer):
    def __init__(self, output_len, **kwargs):
        self.output_len = output_len
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for keras
        :param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim)
        :return:2D tensor with shape (batch_size, output_len)
        """
        hidden_size = int(hidden_states.shape[2])

        score_first_part = Dense(hidden_size,
                                 use_bias=False,
                                 name='attention_socre_vec')(hidden_states)

        h_t = Lambda(lambda x: x[:, -1, :],
                     output_shape=(hidden_size,),
                     name='last_hidden_state')(hidden_states)

        score = dot([score_first_part, h_t], [2, 1], name='attention_socre')

        attention_weights = Activation('softmax', name='attention_weight')(score)

        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')

        attention_vector = Dense(self.output_len, use_bias=False,
                                 activation='tanh',
                                 name='attention_vector')(pre_activation)

        return attention_vector


class RnnAtt(Model):
    def __init__(self,
                 max_len,
                 max_features,
                 embedding_dims,
                 rnn_out_dims=100,
                 att_out_dims=128,
                 dropout_ratio=0.5,
                 class_num=2,
                 last_activation='sigmoid'):

        super(RnnAtt, self).__init__()
        self.embedding_dims = embedding_dims
        self.rnn_out_dims = rnn_out_dims
        self.att_out_dims = att_out_dims

        self.max_len = max_len
        self.max_features = max_features
        self.class_num = class_num
        self.last_activation = last_activation

        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)
        self.dropout = Dropout(dropout_ratio)
        self.lstm = LSTM(units=self.rnn_out_dims, return_sequences=True)
        self.attention = Attention(self.att_out_dims)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        """
        Model of LSTM with Attention for keras
        :param inputs: 3D tensor with shape (batch_size, time_steps, input_dim)
        :return: outputs: 2D tensor with shape (batch_size, probability)
        """
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextCNN_topic must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.max_len:
            raise ValueError(
                'The maxlen of inputs of TextCNN_topic must be %d, but now is %d' % (
                self.maxlen, inputs.get_shape()[1]))

        embedding = self.embedding(inputs)
        seq_in = self.dropout(embedding)
        seq = self.lstm(seq_in)
        att = self.attention(seq)
        att_out = self.dropout(att)
        output = self.classifier(att_out)

        return output


if __name__ == '__main__':
    from keras.datasets import imdb
    from keras.preprocessing import sequence
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

    x_train = sequence.pad_sequences(x_train, maxlen=100)
    x_test = sequence.pad_sequences(x_test, maxlen=100)

    model = RnnAtt()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=10, batch_size=64)


