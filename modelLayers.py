import tensorflow as tf
from keras import layers, losses, models
from server import char2num, num2char, reduceJoin


class BaseBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 3, 3), down_sample=True):
        super(BaseBlock, self).__init__()
        self.layer_names = []
        self.down_sample = down_sample
        layer_collection, self.identity_layer = resLayers(
            filters, kernel_size=kernel_size, down_sample=down_sample
        )
        for layer in layer_collection:
            self.layer_names.append(
                f'layer_{len(self.layer_names) + 1}'
            )
            self.__setattr__(self.layer_names[-1], layer)

    def call(self, x):

        if self.down_sample:
            identity = self.identity_layer(x)
        else:
            identity = x

        for layer_name in self.layer_names:
            curr_layer = self.__getattribute__(layer_name)
            if str(type(curr_layer)).split('.')[-1][:-2] == 'Add':
                x = curr_layer([x, identity])
                continue
            x = curr_layer(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 3, 3), down_sample=True):
        super(ResidualBlock, self).__init__()
        self.hidden_layer_1 = BaseBlock(
            filters, kernel_size=kernel_size, down_sample=down_sample
        )
        self.hidden_layer_2 = BaseBlock(
            filters, kernel_size=kernel_size, down_sample=False
        )

    def call(self, x):
        x = self.hidden_layer_1(x)
        return self.hidden_layer_2(x)


class CTCLoss(losses.Loss):
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
        ),
        layers.Add(),
        layers.BatchNormalization(),
        layers.Activation(
            activation='relu'
        )
    ]
    identity_layer = layers.Conv3D(
        filters=filters, kernel_size=(1, 1, 1), strides=strides, padding='same'
    )

    return layer_collection, identity_layer


def configModel(input_shape):
    model = models.Sequential()

    model.add(
        layers.Conv3D(
            filters=64, kernel_size=(1, 7, 7), padding='same', strides=(1, 1, 2), input_shape=input_shape[1:]
        )
    )

    model.add(
        layers.BatchNormalization()
    )

    model.add(
        layers.Activation(
            activation='relu'
        )
    )

    model.add(
        ResidualBlock(filters=64, kernel_size=(1, 3, 3), down_sample=False)
    )

    model.add(
        ResidualBlock(filters=128, kernel_size=(1, 3, 3), down_sample=True)
    )

    model.add(
        ResidualBlock(filters=256, kernel_size=(1, 3, 3), down_sample=True)
    )

    model.add(
        ResidualBlock(filters=512, kernel_size=(1, 3, 3), down_sample=True)
    )

    model.add(
        layers.AveragePooling3D(
            pool_size=(1, 7, 7), padding='same'
        )
    )
    model.add(
        layers.TimeDistributed(
            layers.Flatten()
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                128, kernel_initializer='Orthogonal', return_sequences=True
            )
        )
    )
    model.add(
        layers.Dropout(0.5)
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                128, kernel_initializer='Orthogonal', return_sequences=True
            )
        )
    )
    model.add(
        layers.Dropout(0.5)
    )
    model.add(
        layers.Dense(
            char2num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'
        )
    )
    return model
