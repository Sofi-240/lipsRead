from server import createPipeline
import tensorflow as tf
from modelLayers import configModel


data = createPipeline()
train = data.take(450)
test = data.skip(450)

sample = iter(data)

X, y = next(sample)

input_shape = X.shape

model = configModel(input_shape)

model.summary()


