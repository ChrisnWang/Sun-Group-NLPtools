import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D, GlobalMaxPooling1D


class RCNN(Model):
    def __init__(self,
                 max_len=100,
                 max_features=5000,
                 embedding_dims=32,
                 rnn_out_dims=100,
                 conv_size=50,
                 dropout_ratio=0.5,
                 class_num=2,
                 last_activation='sigmoid'):
        super(RCNN, self).__init__()
        self.max_len = max_len
        self.max_features = max_features
        self.class_num = class_num
        self.last_activation = last_activation
        self.rnn_out_dims = rnn_out_dims
        self.embedding_dims = embedding_dims
        self.conv_size = conv_size
        self.dropout_ratio = dropout_ratio

        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)
        self.forward_rnn = SimpleRNN(self.rnn_out_dims,
                                     return_sequences=True, dropout=self.dropout_ratio)
        self.backward_rnn = SimpleRNN(self.rnn_out_dims, return_sequences=True,
                                      go_backwards=True, dropout=self.dropout_ratio)

        self.reverse = Lambda(lambda x: tf.reverse(x, axis=[1]))
        self.concatenate = Concatenate(axis=2)

        self.conv = Conv1D(self.conv_size, kernel_size=1, activation='tanh')
        self.max_pooling = GlobalMaxPooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        """
        Model of biRnn with CNN for keras
        :param inputs:4D tensor with shape (batch_size, direction, time_steps, input_dim)
        :return:2D tensor with shape (batch_size, probability)
        """
        if len(inputs) != 3:
            raise ValueError('The length of inputs of RCNN must be 3, but now is %d' % len(inputs))

        input_current = inputs[0]
        input_left = inputs[1]
        input_right = inputs[2]

        if len(input_current.get_shape()) != 2 or len(input_left.get_shape()) != 2 or len(input_right.get_shape()) != 2:
            raise ValueError('The rank of inputs of RCNN must be (2, 2, 2), but now is (%d, %d, %d)'
                             % (len(input_current.get_shape()), len(input_left.get_shape()), len(input_right.get_shape())))
        if input_current.get_shape()[1] != self.max_len or input_left.get_shape()[1] != self.max_len or input_right.get_shape()[1] != self.max_len:
            raise ValueError('The maxlen of inputs of RCNN must be (%d, %d, %d), but now is (%d, %d, %d)'
                             % (self.max_len, self.max_len, self.max_len, input_current.get_shape()[1], input_left.get_shape()[1], input_right.get_shape()[1]))

        embedding_current = self.embedding(input_current)
        embedding_left = self.embedding(input_left)
        embedding_right = self.embedding(input_right)

        x_left = self.forward_rnn(embedding_left)
        x_right = self.backward_rnn(embedding_right)
        x_right = self.reverse(x_right)

        x = self.concatenate([x_left, embedding_current, x_right])
        x = self.conv(x)
        x = self.max_pooling(x)
        output = self.classifier(x)

        return output


def gen_data_rcnn(train, test):
    train_current = train
    train_left = np.hstack([np.expand_dims(train[:, 0], axis=1), train[:, 0:-1]])
    train_right = np.hstack([train[:, 1:], np.expand_dims(train[:, -1], axis=1)])

    test_current = test
    test_left = np.hstack([np.expand_dims(test[:, 0], axis=1), test[:, 0:-1]])
    test_right = np.hstack([test[:, 1:], np.expand_dims(test[:, -1], axis=1)])

    x_tr = [train_current, train_left, train_right]
    x_te = [test_current, test_left, test_right]

    return x_tr, x_te


if __name__ == '__main__':
    from keras.datasets import imdb
    from keras.preprocessing import sequence
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

    x_train = sequence.pad_sequences(x_train, maxlen=100)
    x_test = sequence.pad_sequences(x_test, maxlen=100)

    x_train_current = x_train
    x_train_left = np.hstack([np.expand_dims(x_train[:, 0], axis=1), x_train[:, 0:-1]])
    x_train_right = np.hstack([x_train[:, 1:], np.expand_dims(x_train[:, -1], axis=1)])
    x_test_current = x_test
    x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_test[:, 0:-1]])
    x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])

    model = RCNN()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([x_train_current, x_train_left, x_train_right], y_train,
              validation_data=([x_test_current, x_test_left, x_test_right], y_test),
              epochs=10, batch_size=64)
