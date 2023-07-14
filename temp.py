from server import createPipeline
from modelLayers import PreprocessingLayer, LipsReadModel
import os
import tensorflow as tf
from keras.layers import Input
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")


data = createPipeline()

sample = iter(data)

X, _ = sample.next()


prep = PreprocessingLayer(height=56, width=112)

out = prep(X)

_, ax = plt.subplots(1, 2)
ax[0].imshow(out[0][50].numpy(), cmap='gray')
ax[1].imshow(out[1][50].numpy(), cmap='gray')

input_shape = data.as_numpy_iterator().next()[0][0].shape
model = LipsReadModel(input_shape=input_shape, res_net_layers=10)
model.summary()