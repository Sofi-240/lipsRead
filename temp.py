from server import char2num, num2char, createPipeline, reduceJoin, loadVideo
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from keras import layers, models
import cv2

matplotlib.use("Qt5Agg")


# frames = loadVideo('.\\data\\s1\\brwn2p.mpg')


data = createPipeline()

train = data.take(450)
test = data.skip(450)

sample = iter(data)

val = sample.next()

frames = val[0][0].numpy()


plt.imshow(frames[55, :, :, 0].astype(np.uint8), cmap='gray')


