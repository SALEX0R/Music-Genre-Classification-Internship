
from math import ceil

''' Classifier '''
PREFERRED_MODEL = ".lstm"

''' Dataset '''
# General
DATASET_ROOT = r"/home/felix/Documents/GTZAN/Data"
DATASET_AUDIO_ROOT = r"/home/felix/Documents/GTZAN/Data/genres_original/"
DATASET_SPECTROGRAM_ROOT = r"/home/felix/Documents/GTZAN/Data/images_original/"
# Audio
FULL_LENGTH = 30
DURATION = 3
SAMPLE_RATE = 22050
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = FULL_LENGTH // DURATION
SAMPLES_PER_SEGMENT = (SAMPLE_RATE * FULL_LENGTH) // NUM_SEGMENTS
VALID_SIZE = ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)

''' Model '''
LEARNING_RATE = 0.001
TRAIN_TEST_RATIO = 0.75
LOSS = "sparse_categorical_crossentropy"
BATCH_SIZE = 12
EPOCHS = 100
