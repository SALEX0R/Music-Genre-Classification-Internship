
from config import *
from dataset import GTZAN

import numpy as np
import tensorflow as tf

gt = GTZAN()
data = gt.load_pickle("gtzan.pickle", 0.8)
x_train, y_train, x_test, y_test = data

input_shape = x_train.shape[1], x_train.shape[2]

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(96, input_shape=input_shape, return_sequences=True))
model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    x_train, y_train, 
    validation_data=(x_test, y_test),
    batch_size=12, 
    epochs=100, 
    verbose=2
)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

accuracy = np.sum(y_pred == y_test) / len(y_pred)
print(f"Test accuracy: {accuracy}")

model.save("gt_lstm.h5")
