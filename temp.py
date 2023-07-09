from server import createPipeline, char2num, num2char, reduceJoin
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from keras import layers, models, Input
from modelLayers import ResidualBlock

matplotlib.use("Qt5Agg")

data = createPipeline()
train = data.take(450)
test = data.skip(450)

sample = iter(data)

X, y = next(sample)

input_shape = X.shape

# X = Input(
#     shape=(75, 50, 100, 1), batch_size=1
# )
#
# input_shape = X.shape

model = models.Sequential()

model.add(
    layers.Conv3D(
        filters=64, kernel_size=(1, 7, 7), padding='same', strides=(1, 2, 2), input_shape=input_shape[1:]
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
    layers.Dense(
        char2num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'
    )
)

model.summary()

# out = model(X)