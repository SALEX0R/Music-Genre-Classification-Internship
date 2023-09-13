[comment]: # (Open at https://jbt.github.io/markdown-editor/#dataset-setup for best viewing experience)

## Setup

Here are the instructions on how to set up this project:

1. Firstly, you need to install all the required python modules. To do this, run `pip install -r requirements.txt` in the root of the project directory. This command should install all the required modules.

2. Next up, you need to have a trained model in the backend to classify files. This project comes with a pre-trained model by default. In case you want to train a model yourself, Firstly, go to [this](#dataset-setup) section that talks about creating and integrating a dataset with this project, Then go to `./core` and run `python train.py`. This will initialize and train the model for you and save the trained model in the model's dedicated directory. You can tweak the hyperparameters using the `configuration.py` file. If you still want more control. The entire code to train and handle the model is located in the `__init__.py` file for all the models. Use that.

3. You should be ready to use this software now. Go to the root of the project's directory, then run `python main.py`. If everything works as expected, it should classify all '.wav' files without any errors.

## Dataset Setup

This project uses the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset to train ML models to predict the genre. If you want to train a model yourself. You have to make some changes yourself.  

You have two choices:

1. **Edit the configuration file** - This file is located at `./core/configuration.py` of the project. It contains the variables `DATASET_ROOT` and `DATASET_AUDIO_ROOT`. You must edit these to the location of the dataset (and subfolders) on your system.

2. **Relocate the Dataset and Edit the Dataset Folder Names** - Firstly, extract the archive and change the root folder name from 'archive' to 'GTZAN'. Then, copy/cut the dataset in the same directory as the project. Meaning, `./apollo` and `./GTZAN` should be in the same directory.

## Codebase

This program uses `Qt` for the UI and `Tensorflow` for running the ML model in the background.  

`main.py` - This is the starting point of the application. This script initializes the logger and starts the Qt Application. Very generic starter Qt code.  

`app.py` - This file defines the 'MainWindow' class for this program. This class sets up the UI for the main window and starts a new (Qt) thread for the classifier.  

`./core` - The code in this directory handles everything related to the main 'classifier'. It contains everything from ML models, to scripts that can be used to train them.

`./core/configuration.py` - This file defines all the constants and configuration variables in one place.  

`./core/apollo.py` - This file defines a dummy class to use the ML model. It inherits from `QtCore.QObject` so that we can run the models on a different thread than the application. Also, it can use any model that the user chooses to configure in the `configuration.py` file.

`./core/train.py` - This small script imports all the available models and trains them and saves them. All the code required to train the models is present in the model's dedicated class itself.

`./core/test.py` - This script tests all the available models and prints the accuracy, precision, recall and f1 score for each model.

`./core/model/` - Every (different) ML model will get a dedicated directory in the `core` folder. It will have two files. 

1. `__init__.py` - This script will be called whenever anyone tries to import this model (`./core/apollo.py`). This script defines a class for handling its model. It will contain methods for training, predicting new samples given a filename, loading a pre-trained model, etc.
2. `model.h5` - This will store the weights of the pre-trained model. 

`./core/lstm` and `./core/cnn` are the two models currently integrated with this project.

`./core/lstm` - A very basic LSTM. That uses MFCC features (extracted using Librosa) to classify audio files.

`./core/cnn` - Mid-sized CNN that predicts music genre using the spectrogram generated using the given audio samples.
