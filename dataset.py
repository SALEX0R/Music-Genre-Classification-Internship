
from config import *

import numpy as np
import math
import os
import pickle
import librosa
from sklearn.model_selection import train_test_split

class GTZAN():

    def __init__(self):
        pass

    def extract_features(self, duration=10): # Load training and Testing data

        os.chdir(DATASET_AUDIO_ROOT)

        num_segments = FULL_LENGTH // duration
        samples_per_segment = (SAMPLE_RATE * 30) // num_segments

        features, labels = [], []
        label = 0

        for genre in os.listdir("./"):
            
            print(f"Loading '{genre}'")

            # Process audio files for this genre
            os.chdir(genre)
            
            for filename in os.listdir("./"):

                try:
                    y, _ = librosa.load(filename, sr=SAMPLE_RATE)
                except:
                    print(f"Failed for '{filename}'")
                    continue
            
                for i in range(num_segments):
                    mfcc = librosa.feature.mfcc(
                        y=y[(samples_per_segment*i):(samples_per_segment*(i+1))],
                        sr=SAMPLE_RATE,
                        n_mfcc=N_MFCC,
                        n_fft=N_FFT,
                        hop_length=HOP_LENGTH
                    )
                    mfcc = mfcc.T
                    if len(mfcc) == math.ceil(samples_per_segment / HOP_LENGTH):
                        features.append(mfcc.tolist())
                        labels.append(label)
                
            os.chdir("..")
            label += 1

        # Convert
        x, y = np.array(features), np.array(labels)
        return (x, y)

    def load_pickle(self, filename, train_test_ratio=0.75):
        # Collect data
        with open(filename, "rb") as fp:
            x, y = pickle.load(fp)
        # Split
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, random_state=42)
        return (x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    
    root = os.getcwd()

    # Test    
    dataset = GTZAN()
    data = dataset.extract_features()

    os.chdir(root)
    with open("gtzan.pickle", "wb") as fp:
        pickle.dump(data, fp)

    print(f"Saved dataset in '{os.getcwd()+os.path.sep}gtzan.pickle'")
