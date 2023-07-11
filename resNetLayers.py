import tensorflow as tf
from keras import layers, Input
from server import char2num


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, down_sample=True):
        super(ResnetBlock, self).__init__()
        self.__filters = filters
        self.__down_sample = down_sample
        self.__kernel_size = (1, 3, 3)
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


class ModelResNet18(tf.keras.models.Model):
    def __init__(self, input_shape):
        super(ModelResNet18, self).__init__()
        self.input_layer = Input(shape=input_shape)
        self.layers_names = []

        self.block1_conv = layers.Conv3D(
            filters=64, kernel_size=(1, 7, 7), padding='same', strides=(1, 1, 2),
            input_shape=input_shape, kernel_initializer="he_normal"
        )
        self.block1_bn = layers.BatchNormalization()
        self.block1_pool = layers.MaxPool3D(
            pool_size=(1, 3, 3), padding='same', strides=(1, 1, 1)
        )
        self.layers_names += [
            'block1_conv', 'block1_bn', 'block1_pool'
        ]

        self.block2_1 = ResnetBlock(
            filters=64, down_sample=False
        )
        self.block2_2 = ResnetBlock(
            filters=64, down_sample=False
        )
        self.layers_names += [
            'block2_1', 'block2_2'
        ]

        self.block3_1 = ResnetBlock(
            filters=128, down_sample=True
        )
        self.layers_names += [
            'block3_1', 'block3_2'
        ]

        self.block4_1 = ResnetBlock(
            filters=256, down_sample=True
        )
        self.layers_names += [
            'block4_1', 'block4_2'
        ]

        self.block5_1 = ResnetBlock(
            filters=512, down_sample=True
        )
        self.layers_names += [
            'block5_1', 'block5_2'
        ]

        self.block6_avg = layers.AveragePooling3D(
            pool_size=(1, 7, 7), padding='same'
        )
        self.block6_flt = layers.TimeDistributed(
            layers.Flatten()
        )
        self.block6_lstm = layers.Bidirectional(
            layers.LSTM(
                128, kernel_initializer='Orthogonal', return_sequences=True
            )
        )
        self.block6_drop = layers.Dropout(0.5)
        self.layers_names += [
            'block6_avg', 'block6_flt', 'block6_lstm', 'block6_drop'
        ]

        self.block7_lstm = layers.Bidirectional(
            layers.LSTM(
                128, kernel_initializer='Orthogonal', return_sequences=True
            )
        )
        self.block7_drop = layers.Dropout(0.5)
        self.block7_dense = layers.Dense(
            char2num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'
        )

        self.layers_names += [
            'block7_lstm', 'block7_drop', 'block7_dense'
        ]

        self.output_layer = self.call(self.input_layer)

        super(ModelResNet18, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer_name in self.layers_names:
            x = self.__getattribute__(layer_name)(x)
        return x