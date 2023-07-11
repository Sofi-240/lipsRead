import tensorflow as tf
from keras import layers, Input
from server import char2num, num2char, reduceJoin


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


class ModelAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super(ModelAccuracy, self).__init__()

    def update_state(self, y_true, y_pred):
        return

    def reset_state(self):
        return

    def result(self):
        return


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


class ModelCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
