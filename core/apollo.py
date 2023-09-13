
import os
from . import configuration
import importlib

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

class Classifier(QtCore.QObject):

    classified_signal = QtCore.pyqtSignal(tuple)
    error_encountered_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, control):
        super().__init__()
        module = importlib.import_module(configuration.PREFERRED_MODEL, package="core")
        self.model = module.Model(configuration, control)

    @QtCore.pyqtSlot(tuple)
    def classify(self, details):
        try:
            path, filename = details
            genre = self.model.predict( os.path.join(path, filename) )
            self.classified_signal.emit((path, filename, genre))
        except Exception as err:
            self.error_encountered_signal.emit((details, err))
