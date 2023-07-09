import tensorflow as tf
import numpy as np
from keras import layers, models, Input


def resLayers(filters, kernel_size=(1, 3, 3), down_sample=True):
    if down_sample:
        strides = (1, 2, 2)
    else:
        strides = (1, 1, 1)

    layer_collection = [
        layers.Conv3D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding='same'
        ),
        layers.BatchNormalization(),
        layers.Activation(
            activation='relu'
        ),
        layers.Conv3D(
            filters=filters, kernel_size=kernel_size, strides=(1, 1, 1), padding='same'
        )
    ]

    if down_sample:
        layer_collection += [
            layers.Conv3D(
                filters=filters, kernel_size=(1, 1, 1), strides=strides, padding='same'
            )
        ]

    layer_collection += [
        layers.Add(),
        layers.BatchNormalization(),
        layers.Activation(
            activation='relu'
        )
    ]

    return layer_collection


class BaseBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 3, 3), down_sample=True):
        super(BaseBlock, self).__init__()
        self.layer_names = []
        self.down_sample = down_sample
        for layer in resLayers(filters, kernel_size=kernel_size, down_sample=down_sample):
            self.layer_names.append(
                f'layer_{len(self.layer_names) + 1}'
            )
            self.__setattr__(self.layer_names[-1], layer)

    def call(self, x):

        if self.down_sample:
            iter_layers = self.layer_names[:-4]
            skip = self.__getattribute__(self.layer_names[-4])(x)
        else:
            iter_layers = self.layer_names[:-3]
            skip = x

        for layer_name in iter_layers:
            x = self.__getattribute__(layer_name)(x)

        x = self.__getattribute__(self.layer_names[-3])([x, skip])

        for layer_name in self.layer_names[-2:]:
            x = self.__getattribute__(layer_name)(x)

        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 3, 3), down_sample=True):
        super(ResidualBlock, self).__init__()
        self.hidden_layer_1 = BaseBlock(filters, kernel_size=kernel_size, down_sample=down_sample)
        self.hidden_layer_2 = BaseBlock(filters, kernel_size=kernel_size, down_sample=False)

    def call(self, x):
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        return x

