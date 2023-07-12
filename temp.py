from server import createPipeline, num2char
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from modelLayers import ModelLipRead, CTCLoss, ModelCallback, scheduler, ModelResNet
import os
import tensorflow as tf


data = createPipeline()

train = data.take(450)
validation = data.skip(450)


input_shape = data.as_numpy_iterator().next()[0][0].shape


model = ModelResNet(input_shape)

model.summary()


# model.compile(
#     optimizer=Adam(learning_rate=0.001), loss=CTCLoss()
# )
#
# checkpoint_callback = ModelCheckpoint(
#     os.path.join('models', 'checkpoint'), monitor='loss', save_weights_only=True
# )
#
# schedule_callback = LearningRateScheduler(
#     scheduler
# )

# test_data = iter(data)
# sample = test_data.next()
# yhat = model.predict(sample[0])

# model.load_weights('data\\models\\checkpoint')
#
# test_data = iter(data)
#
# sample = test_data.next()
#
# yhat = model.history.model.predict(sample[0])


# model.fit(
#     train, validation_data=validation, epochs=1,
#     callbacks=[checkpoint_callback, schedule_callback, ModelCallback()]
# )
