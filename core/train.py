
import os
import logging
import configuration
import tensorflow as tf

local_dir = os.getcwd()

''' Logging '''

logging.basicConfig(
    format='''%(asctime)s [%(levelname)s] %(message)s''',
    datefmt="%H:%M:%S",
    filename="training.log", filemode='w',
    encoding="utf-8",
    level=logging.DEBUG
)


''' LSTM '''

import lstm

permission = input("Do you want to train the LSTM model? (y/n):")

if permission == 'y':
    lstm_model = lstm.Model(configuration)
    lstm_model.train()
    os.chdir(local_dir)
    lstm_model.model.save("./lstm/lstm.h5")

''' CNN '''

import cnn

permission = input("Do you want to train the CNN model? (y/n):")

if permission == 'y':
    cnn_model = cnn.Model(configuration)
    cnn_model.train()
    os.chdir(local_dir)
    cnn_model.model.save("./cnn/cnn.h5")
