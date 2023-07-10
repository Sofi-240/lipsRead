from server import createPipeline
import tensorflow as tf
from modelLayers import configModel, ModelLipNet


# data = createPipeline()
# train = data.take(450)
# test = data.skip(450)
#
# sample = iter(data)
#
# X, y = next(sample)
#
# input_shape = X.shape

input_shape = (2, 75, 56, 112, 1)

model = ModelLipNet(input_shape)

model.summary()


