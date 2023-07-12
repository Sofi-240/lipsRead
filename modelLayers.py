import tensorflow as tf
from keras import layers, Input
from server import char2num, num2char
import pandas as pd
from fuzzywuzzy import fuzz


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.loss_function = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_size = tf.cast(
            tf.shape(y_true)[0], dtype='int64'
        )
        input_size = tf.cast(
            tf.shape(y_pred)[1], dtype='int64'
        )
        input_size = input_size * tf.ones(
            shape=(batch_size, 1), dtype='int64'
        )

        label_size = tf.cast(
            tf.shape(y_true)[1], dtype='int64'
        )
        label_size = label_size * tf.ones(
            shape=(batch_size, 1), dtype='int64'
        )

        loss = self.loss_function(
            y_true=y_true, y_pred=y_pred, input_length=input_size, label_length=label_size
        )
        return loss


class FuzzySimilarity(tf.keras.metrics.Metric):
    def __init__(self, name='FuzzySimilarity', **kwargs):
        super(FuzzySimilarity, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(
            name="total", initializer="zeros"
        )
        self.count = self.add_weight(
            name="count", initializer="zeros"
        )

    @staticmethod
    def update_state_np(y_true, y_pred):
        decoded = tf.keras.backend.ctc_decode(
            y_pred, [y_pred.shape[1]] * y_pred.shape[0], greedy=False
        )[0][0].numpy()
        y_true_str = [
            tf.strings.reduce_join(num2char(y)).numpy().decode('utf-8') for y in y_true
        ]
        y_pred_str = [
            tf.strings.reduce_join(num2char(y)).numpy().decode('utf-8') for y in decoded
        ]
        sim = [
            fuzz.ratio(yt, yp) / 100 for yt, yp in zip(y_true_str, y_pred_str)
        ]
        return sum(sim) / 2

    def update_state(self, y_true, y_pred, sample_weight=None):
        sim = tf.py_function(
            self.update_state_np, [y_true, y_pred], tf.float32
        )
        self.total.assign_add(tf.cast(
            sim, self._dtype
        ))
        self.count.assign_add(
            tf.cast(
                1, self._dtype
            )
        )

    def result(self):
        return tf.math.divide(self.total, self.count)


class ModelLipRead(tf.keras.models.Model):
    def __init__(self, input_shape):
        super(ModelLipRead, self).__init__()
        self.input_layer = Input(shape=input_shape)
        self.layers_names = []

        for i, filters_size in enumerate([128, 256, 75]):
            names = [
                f'{name}_{i + 1}' for name in ['conv', 'act', 'pool']
            ]
            self.__setattr__(
                names[0],
                layers.Conv3D(
                    filters=filters_size, kernel_size=(3, 3, 3),
                    padding='same', strides=(1, 1, 1),
                    kernel_initializer="he_normal", name=names[0]
                )
            )
            self.__setattr__(
                names[1],
                layers.Activation(
                    activation='relu',
                    name=names[1]
                )
            )
            self.__setattr__(
                names[2],
                layers.MaxPool3D(
                    pool_size=(1, 2, 2), padding='valid',
                    name=names[2]
                )
            )
            self.layers_names += names

        self.flt = layers.TimeDistributed(
            layers.Flatten(), name='flt'
        )
        self.layers_names += [
            'flt'
        ]

        self.lstm_1 = layers.Bidirectional(
            layers.LSTM(
                128, kernel_initializer='Orthogonal', return_sequences=True
            ), name='lstm_1'
        )
        self.drop_1 = layers.Dropout(0.5, name='drop_1')
        self.layers_names += [
            'lstm_1', 'drop_1'
        ]

        self.lstm_2 = layers.Bidirectional(
            layers.LSTM(
                128, kernel_initializer='Orthogonal', return_sequences=True,
            ), name='lstm_2'
        )
        self.drop_2 = layers.Dropout(0.5, name='drop_2')
        self.layers_names += [
            'lstm_2', 'drop_2'
        ]

        self.dense = layers.Dense(
            char2num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax',
            name='dense'
        )
        self.layers_names += [
            'dense'
        ]

        self.output_layer = self.call(self.input_layer)
        super(ModelLipRead, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, x, training=False):

        for layer_name in self.layers_names:
            x = self.__getattribute__(layer_name)(x)

        return x


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, down_sample=True, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.__filters = filters
        self.__down_sample = down_sample
        self.__kernel_size = (3, 3, 3)
        self.__strides = [(1, 2, 2), (1, 1, 1)] if down_sample else [(1, 1, 1), (1, 1, 1)]
        self.__kernel_initializer = "he_normal"

        self.identity_layers_names = []
        self.block_layers_names = []

        self.conv_1 = layers.Conv3D(
            filters=self.__filters, kernel_size=self.__kernel_size,
            strides=self.__strides[0], padding='same',
            kernel_initializer=self.__kernel_initializer
        )
        self.bn_1 = layers.BatchNormalization()
        self.act_1 = layers.Activation(
            activation='relu'
        )
        self.block_layers_names += [
            'conv_1', 'bn_1', 'act_1'
        ]

        self.conv_2 = layers.Conv3D(
            filters=self.__filters, kernel_size=self.__kernel_size,
            strides=self.__strides[1], padding='same',
            kernel_initializer=self.__kernel_initializer
        )
        self.bn_2 = layers.BatchNormalization()
        self.block_layers_names += [
            'conv_2', 'bn_2',
        ]

        self.marge = layers.Add()
        self.out = layers.Activation(
            activation='relu'
        )
        self.block_layers_names += [
            'marge', 'out',
        ]

        if self.__down_sample:
            self.identity_conv = layers.Conv3D(
                filters=self.__filters, kernel_size=(1, 1, 1),
                strides=self.__strides[0], padding='same',
                kernel_initializer=self.__kernel_initializer
            )
            self.identity_bn = layers.BatchNormalization()
            self.identity_layers_names += [
                'identity_conv', 'identity_bn'
            ]

    def call(self, x):
        identity = x
        for layer_name in self.identity_layers_names:
            identity = self.__getattribute__(layer_name)(identity)

        for layer_name in self.block_layers_names:
            if layer_name == 'marge':
                x = self.__getattribute__(layer_name)([identity, x])
                continue
            x = self.__getattribute__(layer_name)(x)

        return x


class ModelResNet(tf.keras.models.Model):
    def __init__(self, input_shape, res_net_layers=10, **kwargs):
        super(ModelResNet, self).__init__(**kwargs)
        self.input_layer = Input(shape=input_shape, name='Input')
        self.layers_names = []

        self.conv = layers.Conv3D(
            filters=64, kernel_size=(1, 7, 7), padding='same', strides=(1, 1, 2),
            input_shape=input_shape, kernel_initializer="he_normal", name='conv'
        )
        self.bn = layers.BatchNormalization(name='bn')
        self.max_pool = layers.MaxPool3D(
            pool_size=(1, 3, 3), padding='same', strides=(1, 1, 1), name='max_pool'
        )
        self.layers_names += [
            'conv', 'bn', 'max_pool'
        ]
        down_sample_cond = [False] if res_net_layers == 10 else [False, False]

        for block_n, filters in enumerate([64, 128, 256, 512]):

            for i, sample in enumerate(down_sample_cond):
                name = f'block{block_n + 1}_{i + 1}'
                self.__setattr__(
                    name,
                    ResnetBlock(
                        filters=filters, down_sample=sample, name=name
                    )
                )
                self.layers_names += [name]
            down_sample_cond[0] = True

        self.avg = layers.AveragePooling3D(
            pool_size=(1, 7, 7), padding='same', name='avg'
        )
        self.flatten = layers.TimeDistributed(
            layers.Flatten(), name='flatten'
        )
        self.layers_names += [
            'avg', 'flatten'
        ]

        for i in range(2):
            names = [f'lstm_{i + 1}', f'drop_{i + 1}']
            self.__setattr__(
                names[0],
                layers.Bidirectional(
                    layers.LSTM(
                        128, kernel_initializer='Orthogonal', return_sequences=True
                    ), name=names[0]
                )
            )
            self.__setattr__(
                names[1],
                layers.Dropout(
                    0.5, name=names[1]
                )
            )
            self.layers_names += names

        self.dense = layers.Dense(
            char2num.vocabulary_size() + 1, kernel_initializer='he_normal',
            activation='softmax', name='dense'
        )

        self.layers_names += [
            'dense'
        ]

        self.output_layer = self.call(self.input_layer)

        super(ModelResNet, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer_name in self.layers_names:
            x = self.__getattribute__(layer_name)(x)
        return x


class ModelCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
