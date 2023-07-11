from server import createPipeline, num2char, char2num, reduceJoin
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from modelLayers import ModelLipRead, CTCLoss, ModelCallback, scheduler
import os


data = createPipeline()

train = data.take(10)
test = data.skip(10)
test = test.take(10)

input_shape = data.as_numpy_iterator().next()[0][0].shape

model = ModelLipRead(input_shape)

model.summary()

# model.compile(
#     optimizer=Adam(learning_rate=0.0001), loss=CTCLoss()
# )
#
# checkpoint_callback = ModelCheckpoint(
#     os.path.join('models', 'checkpoint'), monitor='loss', save_weights_only=True
# )
#
# schedule_callback = LearningRateScheduler(
#     scheduler
# )
#
# model.fit(
#     train, validation_data=test, epochs=1,
#     callbacks=[checkpoint_callback, schedule_callback, ModelCallback()]
# )
