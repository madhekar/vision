import tensorflow as tf
import numpy as np

class CustomCallback(tf.keras.callbacks.Callback):
   def __init__(self, data):
        self.data = data
   def on_predict_batch_end(self, batch, logs=None):
        dns_layer = self.model.layers[6]
        outputs = dns_layer(self.data)
        tf.print(f'\n batch: {batch}')
        tf.print(f'\n input: {self.data}')
        tf.print(f'\n output: {outputs}')


x_train = tf.random.normal((10, 32, 32))
y_train = tf.random.uniform((10, 1), maxval=10)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(256))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax')) 
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax')) 
model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(False))
model.summary()    

for layer in model.layers:
    print(layer)

model.fit(x_train, y_train , epochs=1, batch_size=32)
yhat = model.predict(tf.random.normal((5, 32, 32)), batch_size=1, callbacks=[CustomCallback(np.random.rand(1,10))])

