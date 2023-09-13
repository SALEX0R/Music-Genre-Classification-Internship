
import os
from black import lib2to3_parse
import librosa
import logging
import audiofile
import numpy as np
import tensorflow as tf

from math import ceil
from sklearn.model_selection import train_test_split

class Model():

    def __init__(self, configuration, control={"halt":False}):
        self.conf = configuration
        self.model = None
        self.control = control
        self.label_map = sorted("blues classical country disco hiphop jazz metal pop reggae rock".split(' '))
        self.logger = logging.getLogger(__name__)

    def load_model(self, path="./core/cnn/cnn.h5"):
        self.logger.info("Loading pretrained model!")
        self.model = tf.keras.models.load_model(path)

    def load_data(self):

        conf = self.conf
        
        self.logger.info("Loading dataset")

        # Switch to dataset directory
        os.chdir(conf.DATASET_AUDIO_ROOT)

        features, labels = [], []
        label = 0

        genres = sorted(os.listdir("./"))

        for genre in genres:
            
            self.logger.info(f"Loading '{genre}'")
            os.chdir(genre)
            
            for filename in os.listdir("./"):

                try:
                    y, _ = librosa.load(filename, sr=conf.SAMPLE_RATE)
                except Exception as err:
                    self.logger.error(f"Error Encountered in {filename}: {err}")
                    continue

                for i in range(conf.NUM_SEGMENTS):
                    start = conf.SAMPLES_PER_SEGMENT * i
                    end = conf.SAMPLES_PER_SEGMENT * (i+1)
                    section = y[start:end]
                    spectrogram = librosa.feature.melspectrogram(y=section, sr=conf.SAMPLE_RATE)
                    if spectrogram.shape == (128, 130):
                        features.append(spectrogram[:, :, None])
                        labels.append(label)
            
            os.chdir("..")
            label += 1

        # Convert
        x, y = np.array(features), np.array(labels)
        self.logger.info(f"{x.shape[0]} samples collected in total.")
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=conf.TRAIN_TEST_RATIO, random_state=42)
        return (x_train, y_train, x_test, y_test)

    def train(self):

        conf = self.conf
        
        x_train, y_train, x_test, y_test = self.load_data()
        input_shape = x_train[0].shape

        model = tf.keras.Sequential()
        
        model.add(tf.keras.layers.Conv2D(8, (3, 3), activation="relu", input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(10, activation="softmax"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss=conf.LOSS, 
            metrics=["accuracy"]
        )

        model.summary()

        self.logger.info("Training Model")

        model.fit(
            x_train, y_train, 
            validation_data=(x_test, y_test),
            batch_size=64, 
            epochs=300, 
            verbose=2
        )

        self.model = model

    def predict(self, filename):

        if self.control["halt"] == True:
            raise Exception("Stopped Program")

        conf = self.conf
        
        if self.model == None:
            self.load_model()

        y, _ = audiofile.read(filename)
        
        full_length = int(librosa.get_duration(filename=filename))
        num_segments = full_length / conf.DURATION
        samples_per_segment = int((conf.SAMPLE_RATE * full_length) / num_segments)
        valid_size = ceil(samples_per_segment / conf.HOP_LENGTH)

        # Store scores for all classes
        predictions = [0 for _ in range(10)]

        for i in range(int(num_segments)):
        
            start = samples_per_segment * i
            end = samples_per_segment * (i+1)
        
            section = y[start:end]
            spectrogram = librosa.feature.melspectrogram(y=section, sr=conf.SAMPLE_RATE)
            if spectrogram.shape == (128, 130):
                features = np.array([spectrogram[:, :, None]])
                # Predict Genre from MFCC Features
                prediction = self.model.predict(features)[0]
                prediction = prediction > 0.5
                # Update Predictions
                predictions += prediction

        # Get the class with most votes
        class_idx = np.argmax(predictions)
        # Return the Class Name that got most votes
        return self.label_map[class_idx]
