from server import createPipeline
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from keras import layers, models, Input
from modelLayers import ResidualBlock

matplotlib.use("Qt5Agg")


# data = createPipeline()
# train = data.take(450)
# test = data.skip(450)
# input_shape = data.as_numpy_iterator().next()[0][0].shape


X = Input(
    shape=(75, 50, 100, 1), batch_size=2
)



lay = ResidualBlock()










