from server import createPipeline, num2char
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from modelLayers import CTCLoss, ModelCallback, LipsReadModel, FuzzySimilarity, PreprocessingLayer
import os
import tensorflow as tf
from keras.layers import Input

# data = createPipeline()
#
# validation_size = int(0.1 * len(data))
#
# test_size = int(0.1 * len(data))
#
# test = data.take(test_size + validation_size)
# validation = test.take(validation_size)
# test = test.skip(validation_size)
# train = data.skip(test_size + validation_size)
#
#
# prep = ModelPreprocessing(input_shape=(75, 288, 360, 3), height=56, width=112)
#
# sample = next(iter(data))
#
# i = Input(sample[0][0].shape)
# ret = prep.call(i)

# input_shape = data.as_numpy_iterator().next()[0][0].shape
model = LipsReadModel((75, 288, 360, 3))

model.summary()

# model.compile(
#     optimizer=Adam(learning_rate=0.001), loss=CTCLoss(), metrics=[FuzzySimilarity()]
# )
#
# checkpoint_callback = ModelCheckpoint(
#     os.path.join('models', 'checkpoint'), monitor='loss', save_weights_only=True
# )
# sample = iter(test).next()
# y_pred = model.predict(sample[0])
#
# f = FuzzySimilarity()
# f.update_state(sample[1], tf.convert_to_tensor(y_pred))

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
