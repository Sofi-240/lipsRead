from server import createPipeline, num2char
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from modelLayers import CTCLoss, ModelCallback, scheduler, ModelResNet, FuzzySimilarity
import os
import tensorflow as tf


data = createPipeline()

train = data.take(2)
test = data.skip(2)
validation = test.take(2)

input_shape = data.as_numpy_iterator().next()[0][0].shape


model = ModelResNet(input_shape)

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001), loss=CTCLoss(), metrics=[FuzzySimilarity()]
)

checkpoint_callback = ModelCheckpoint(
    os.path.join('models', 'checkpoint'), monitor='loss', save_weights_only=True
)
sample = iter(test).next()
y_pred = model.predict(sample[0])

f = FuzzySimilarity()
f.update_state(sample[1], tf.convert_to_tensor(y_pred))

# model.load_weights('data\\models\\checkpoint')
#
# test_data = iter(data)
#
# sample = test_data.next()
#
# yhat = model.history.model.predict(sample[0])


# model.fit(
#     train, validation_data=validation, epochs=1,
#     callbacks=[checkpoint_callback, ModelCallback()]
# )
