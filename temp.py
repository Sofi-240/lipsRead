from server import char2num, num2char, createPipeline, reduceJoin, loadVideo
from preprocessing import extractFace
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np
import cv2

matplotlib.use("Qt5Agg")

# frames = loadVideo('.\\data\\s1\\brwn2p.mpg')

data = createPipeline()

train = data.take(450)
test = data.skip(450)

sample = data.as_numpy_iterator()

val = sample.next()

frames = val[0][1]

plt.imshow(frames[4, :, :, :].astype(np.uint8), cmap='gray')

