import tensorflow as tf
from keras import layers, Input
from dataPipeline import char2num, num2char
from fuzzywuzzy import fuzz
import numpy as np
import cv2
import random
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


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
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(
            shape=input_shape[0], dtype='int32'
        ) * tf.cast(
            input_shape[1], 'int32'
        )

        decoded = tf.keras.backend.ctc_decode(
            y_pred, input_length, greedy=False
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
        return sum(sim) / len(sim)

    def update_state(self, y_true, y_pred, sample_weight=None):
        sim = tf.py_function(
            self.update_state_np, [y_true, y_pred], tf.float32
        )
        self.total.assign_add(
            tf.cast(
                sim, self._dtype
            )
        )
        self.count.assign_add(
            tf.cast(
                1, self._dtype
            )
        )

    def result(self):
        return tf.math.divide(self.total, self.count)


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

    def call(self, x, training=False):
        identity = x
        for layer_name in self.identity_layers_names:
            identity = self.__getattribute__(layer_name)(identity)

        for layer_name in self.block_layers_names:
            if layer_name == 'marge':
                x = self.__getattribute__(layer_name)([identity, x])
                continue
            x = self.__getattribute__(layer_name)(x)

        return x


class LipsReadModel(tf.keras.models.Model):
    def __init__(self, input_shape, res_net_layers=10, **kwargs):
        super(LipsReadModel, self).__init__(**kwargs)
        self.input_layer = Input(shape=input_shape, name='Input')
        self.layers_names = []

        self.preprocessing = PreprocessingLayer(
            height=56, width=112, name='preprocessing'
        )

        self.conv = layers.Conv3D(
            filters=64, kernel_size=(1, 7, 7), padding='same', strides=(1, 1, 2),
            kernel_initializer="he_normal", name='conv'
        )
        self.bn = layers.BatchNormalization(name='bn')
        self.max_pool = layers.MaxPool3D(
            pool_size=(1, 3, 3), padding='same', strides=(1, 1, 1), name='max_pool'
        )
        self.layers_names += [
            'preprocessing', 'conv', 'bn', 'max_pool'
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

        super(LipsReadModel, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer
        )

    def call(self, x, training=False):
        for layer_name in self.layers_names:
            x = self.__getattribute__(layer_name)(x)
        return x


class ModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate_base=0.001, warmup_learning_rate=0.0, warmup_steps_practice=0.02,
                 global_step_init=0, fuzzy_patience=5):
        super(ModelCallback, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps_practice = warmup_steps_practice
        self.global_step = global_step_init
        self.total_steps = None
        self.warmup_steps = None
        self.slope = None
        self._built = False
        self.fuzzy_patience = fuzzy_patience
        self._best_fuzzy = 0
        self._fuzzy_patience_wait = 0

    def _build(self):
        self.total_steps = int(
            self.params['epochs'] * self.params['steps']
        )
        self.warmup_steps = int(
            self.total_steps * self.warmup_steps_practice
        )
        self.slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
        self._built = True

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1

    def on_batch_begin(self, batch, logs=None):
        if not self._built:
            self._build()

        if self.total_steps < self.warmup_steps:
            raise ValueError(
                'total_steps must be larger or equal to warmup_steps.'
            )
        if (self.warmup_steps > 0) and (self.learning_rate_base < self.warmup_learning_rate):
            raise ValueError(
                'learning_rate_base must be larger or equal to warmup_learning_rate.'
            )

        lr = 0.5 * self.learning_rate_base * (1 + np.cos(np.pi * (
                (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        )))

        if self.warmup_steps > 0:
            warmup_rate = self.slope * self.global_step + self.warmup_learning_rate
            lr = np.where(
                self.global_step < self.warmup_steps, warmup_rate, lr
            )
        lr = np.where(
            self.global_step > self.total_steps, 0.0, lr
        )
        tf.keras.backend.set_value(
            self.model.optimizer.lr, lr
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = float(
            tf.keras.backend.get_value(self.model.optimizer.lr)
        )
        if logs.get('FuzzySimilarity') is not None:
            fz = logs['FuzzySimilarity']
            self._fuzzy_patience_wait += 1
            if fz > self._best_fuzzy:
                self._best_fuzzy = fz
                self._fuzzy_patience_wait = 0
            if self._fuzzy_patience_wait >= self.fuzzy_patience:
                self.model.stop_training = True


class PreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
        self._out_height = height
        self._out_width = width
        self._out_dim = 1
        self.resize = layers.Resizing(
            height=self._out_height, width=self._out_width,
            interpolation='bicubic', name='resize'
        )
        self.rescale = tf.keras.layers.TimeDistributed(
            layers.Rescaling(
                scale=1. / 255
            ), name='rescale'
        )

    def call(self, x):
        if x.shape[0] is None:
            x = tf.image.rgb_to_grayscale(x)
            x = tf.keras.layers.TimeDistributed(
                self.resize
            )(x)
            return self.rescale(x)

        x = tf.py_function(
            self._mouthSegmentation, [x], x.dtype
        )

        x = self.rescale(x)
        return x

    def _mouthSegmentation(self, frames):
        _type = frames.dtype
        if tf.is_tensor(frames) and frames.shape[0] is not None:
            frames = frames.numpy()
        if frames.shape[0] is None:
            return
        if not isinstance(frames, np.ndarray):
            raise ValueError(
                'frames isinstance need to be tensor or np.ndarray array'
            )

        _init_shape = frames.shape

        if len(_init_shape) < 4:
            raise ValueError(
                'frames shape need to be 4D (without batches) or 5D (with batches)'
            )
        if _init_shape[-1] != 3:
            raise ValueError(
                'frames need to be in RGB format'
            )
        if len(_init_shape) < 5:
            frames = np.expand_dims(
                frames, axis=0
            )

        _init_shape = frames.shape

        bg_crop = []
        face_crop = []
        mouth_crop = np.zeros(
            (*_init_shape[:2], self._out_height, self._out_width, 1)
        )

        frames_median_rgb = np.median(
            frames, axis=1
        )

        for b, image_median in enumerate(frames_median_rgb):
            image_hsv = cv2.cvtColor(
                image_median, cv2.COLOR_RGB2HSV
            )

            threshold, image_bin = cv2.threshold(
                image_hsv[:, :, 0].astype(np.uint8),
                0, 1,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            kernel = np.ones(
                (5, 5), dtype=np.uint8
            )

            image_bin = cv2.morphologyEx(
                image_bin, cv2.MORPH_OPEN, kernel, iterations=1
            )

            points = cv2.findNonZero(image_bin)
            if points is None:
                bg_crop.append(frames[b])
                continue

            rect = cv2.boundingRect(points)

            crop_frames = tf.image.crop_to_bounding_box(
                frames[b],
                offset_width=rect[0], offset_height=rect[1],
                target_width=rect[2], target_height=rect[3]
            )

            crop_frames = tf.image.rgb_to_grayscale(
                crop_frames
            )

            crop_frames = tf.image.adjust_contrast(
                crop_frames, 1
            ).numpy()

            bg_crop.append(
                crop_frames
            )

        sample_index = random.sample(
            list(
                range(bg_crop[0].shape[0])
            ), 4
        )

        for crop_frames in bg_crop:
            crop_median = np.median(
                crop_frames, axis=0
            ).astype(np.uint8)

            face = faceCascade(crop_median)
            face_prop = [
                face[0][1] - face[0][0],
                face[1][1] - face[1][0]
            ]

            for smpl in sample_index:
                curr_frame = crop_frames[smpl]

                curr_face = faceCascade(curr_frame)
                curr_prop = [
                    curr_face[0][1] - curr_face[0][0],
                    curr_face[1][1] - curr_face[1][0]
                ]
                if (curr_face[0][0] > 0 and curr_face[0][1] < curr_frame.shape[1] - 1 and curr_prop[0] > face_prop[
                    0]) or (
                        face_prop[0] == curr_frame.shape[1] - 1):
                    face[0] = curr_face[0]
                    face_prop[0] = curr_prop[0]

                if (curr_face[1][0] > 0 and curr_face[1][1] < curr_frame.shape[0] - 1 and curr_prop[1] > face_prop[
                    1]) or (
                        face_prop[1] == curr_frame.shape[0] - 1):
                    face[1] = curr_face[1]
                    face_prop[1] = curr_prop[1]

            face_crop.append(
                (
                    crop_frames[:, face[1][0]:face[1][1], face[0][0]:face[0][1], :],
                    crop_median[face[1][0]:face[1][1], face[0][0]:face[0][1], :]
                )
            )

        sample_index = random.sample(
            list(
                range(bg_crop[0].shape[0])
            ), 4
        )

        for b, crop in enumerate(face_crop):
            crop_frames, crop_median = crop
            mouth = mouthCascade(crop_median)
            mouth_prop = [
                mouth[0][1] - mouth[0][0],
                mouth[1][1] - mouth[1][0]
            ]

            for smpl in sample_index:
                curr_frame = crop_frames[smpl]

                curr_mouth = mouthCascade(curr_frame)
                curr_prop = [
                    curr_mouth[0][1] - curr_mouth[0][0],
                    curr_mouth[1][1] - curr_mouth[1][0]
                ]

                if (curr_mouth[0][0] > 0 and curr_mouth[0][1] < curr_frame.shape[1] - 1 and curr_prop[0] > mouth_prop[
                    0]) or (
                        mouth_prop[0] == curr_frame.shape[1] - 1):
                    mouth[0] = curr_mouth[0]
                    mouth_prop[0] = curr_prop[0]

                if (curr_mouth[1][0] > curr_frame.shape[0] // 2 and curr_prop[1] > mouth_prop[1]) or (
                        mouth_prop[1] == (curr_frame.shape[0] // 2) - 1):
                    mouth[1] = curr_mouth[1]
                    mouth_prop[1] = curr_prop[1]

            crop_frames = crop_frames[:, mouth[1][0]:mouth[1][1], mouth[0][0]:mouth[0][1], :]
            mouth_crop[b] = tf.cast(
                self.resize(crop_frames), dtype=_type
            )
        return mouth_crop


def faceCascade(img_gray):
    cascade = cv2.CascadeClassifier(
        "data\\haarcascade_frontalface_default.xml"
    )

    rect = cascade.detectMultiScale(
        img_gray.astype(np.uint8)
    )

    if type(rect) == tuple:
        bound = [
            [
                0, img_gray.shape[1] - 1
            ],
            [
                0, img_gray.shape[0] - 1
            ]
        ]
        return bound
    bound = [
        [
            rect[0, 0], min([img_gray.shape[1] - 1, rect[0, 0] + rect[0, 2]])
        ],
        [
            rect[0, 1], min([img_gray.shape[0] - 1, rect[0, 1] + rect[0, 3] + 10])
        ],
    ]
    return bound


def mouthCascade(img_gray):
    cascade = cv2.CascadeClassifier(
        "data\\haarcascade_mcs_mouth.xml"
    )

    rect = cascade.detectMultiScale(
        img_gray.astype(np.uint8), 1.4
    )

    mid = img_gray.shape[0] // 2

    bound = [
        [
            0, img_gray.shape[1] - 1
        ],
        [
            mid, img_gray.shape[0] - 1
        ]
    ]

    if type(rect) == tuple:
        return bound

    rect = rect[rect[:, 1] > mid, :]

    if rect.shape[0] == 0:
        return bound

    if rect.shape[0] > 1:
        indices = np.argsort(rect[:, 2])[::-1]
        rect = np.reshape(
            rect[indices[0], :], (1, 4)
        )

    bound = [
        [
            max([0, rect[0, 0] - 10]), min([img_gray.shape[1] - 1, rect[0, 0] + rect[0, 2] + 10])
        ],
        [
            max([mid, rect[0, 1] - 10]), min([img_gray.shape[0] - 1, rect[0, 1] + rect[0, 3] + 10])
        ],
    ]
    return bound


def buildModel(input_shape):
    model = LipsReadModel(input_shape)
    model.compile(
        optimizer=Adam(),
        loss=CTCLoss(),
        metrics=[FuzzySimilarity()]
    )

    checkpoint_path = "data\\models\\checkpoint\\"

    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_FuzzySimilarity', verbose=1, save_weights_only=True,
        save_freq='epoch', mode='auto', save_best_only=True
    )

    return model, checkpoint_callback
