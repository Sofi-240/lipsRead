import tensorflow as tf
from keras import layers, Input
from server import char2num
import pandas as pd


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


class ModelLipRead(tf.keras.models.Model):
    def __init__(self, input_shape):
        super(ModelLipRead, self).__init__()
        self.input_layer_1 = Input(shape=input_shape)
        self.input_layer_2 = Input(shape=input_shape)
        self.layers_names = []

        for i, filters_size in enumerate([64, 256, 75]):
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
            if i == 0:
                self.marge = layers.Concatenate()
                self.layers_names += ['marge']

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

        self.output_layer = self.call((self.input_layer_1, self.input_layer_2))
        super(ModelLipRead, self).__init__(
            inputs=(self.input_layer_1, self.input_layer_2),
            outputs=self.output_layer
        )

    def call(self, inputs, training=False):
        inputs = list(inputs)
        mrg = False
        x = None
        for layer_name in self.layers_names:
            if mrg:
                x = self.__getattribute__(layer_name)(x)
            elif layer_name == 'marge':
                x = self.__getattribute__(layer_name)(inputs)
                mrg = True
            else:
                lay = self.__getattribute__(layer_name)
                inputs = [lay(inp) for inp in inputs]

        return x


class ModelCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.df = pd.DataFrame(
            columns=[
                'epoch', 'loss', 'lr'
            ]
        )

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = tf.keras.backend.eval(optimizer.lr)
        self.df.loc[epoch, :] = [epoch + 1, logs['loss'], lr]
        print("\nEnd epoch {}| loss: {} | lr: {}".format(epoch, logs['loss'], lr))


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
