
import os
import shutil
import logging
from core.apollo import Classifier

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

class MainWindow(QMainWindow):

    '''
        - Runs as the main controller thread
        - Uses the core to classify audio files
    '''

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("Apollo: Organize Music!")
        self.setFixedHeight(650)
        self.setFixedWidth(400)

        self.central_widget = QtWidgets.QLabel(self)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Setup Banner
        self.banner_holder = QtWidgets.QLabel()
        self.banner = QtGui.QPixmap("./assets/meme.png")
        self.banner_holder.setPixmap(self.banner.scaled(400, 400))

        self.controls = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QHBoxLayout(self.controls)
        ## Setup Buttons
        self.browse_button = QtWidgets.QPushButton(self.controls)
        self.halt_button = QtWidgets.QPushButton(self.controls)
        ### Setup Buttons Style
        self.browse_button.setText("Browse")
        self.browse_button.setStyleSheet("color: white; background-color: #222222;")
        self.halt_button.setText("Halt")
        self.halt_button.setStyleSheet("color: white; background-color: #ff0000;")
        self.browse_button.clicked.connect(self.browse)
        self.halt_button.clicked.connect(self.halt)
        ### Setup Controls Layout 
        self.controls_layout.addWidget(self.browse_button)
        self.controls_layout.addWidget(self.halt_button)
        self.controls_layout.setContentsMargins(5, 0, 5, 0)
        self.controls.setLayout(self.controls_layout)
    
        # Setup Message Box
        font = QtGui.QFont()
        font.setPointSize(11)
        self.message_box = QtWidgets.QListWidget(self)
        self.message_box.setFont(font)

        # Add UI Components to Layout
        self.layout.addWidget(self.banner_holder)
        self.layout.addWidget(self.controls)
        self.layout.addWidget(self.message_box)

        # Configure Layout
        self.layout.setContentsMargins(0, 0, 0, 5)
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Setup Audio Classifier on a different thread
        self.classifier_control = {"halt": False}
        self.classifier_thread = QtCore.QThread()
        self.classifier_worker = Classifier(self.classifier_control)
        self.classifier_worker.moveToThread(self.classifier_thread)
        # Setup communication signals
        self.classifier_worker.classified_signal.connect(self.classify)
        self.classifier_worker.error_encountered_signal.connect(self.error_handler)
        # Start Classifier Thread
        self.classifier_thread.start()

        self.addLog("UI initialization complete.")
        self.addLog("Browse a file or folder!")
        self.show()

    def browse(self):

        self.classifier_control["halt"] = False

        # Get path from user
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Directory", "", QtWidgets.QFileDialog.ShowDirsOnly)

        self.addLog(f"Opening Folder '{path}'")

        for filename in os.listdir(path):
            
            # Make sure we are only processing files
            if not os.path.isfile( os.path.join(path, filename) ):
                continue
            
            # Only files with a valid extensions are processed
            extension = filename.split('.')[-1]
            
            # Only process the extensions supported by 'Librosa'
            if extension in ("wav", "flac", "ogg"):
            
                QtCore.QMetaObject.invokeMethod(
                    self.classifier_worker, 
                    "classify", 
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(tuple, (path, filename))
                )

                self.addLog(f"Added '{filename}' to the queue")   


    def classify(self, details):
        path, filename, genre = details
        if genre not in os.listdir(path):
            os.mkdir(os.path.join(path, genre))
        self.addLog(f"Classified {filename} as {genre}.")
        self.logger.info(f"Classified {filename} as {genre}.")
        shutil.move(os.path.join(path, filename), os.path.join(path, genre, filename))
    
    def error_handler(self, details):
        (_, filename), err = details
        self.addLog(f"Could not classify '{filename}' (see logs for error).")
        self.logger.error(f"Error Encountered: {err}")

    def halt(self):
        self.classifier_control["halt"] = True
        self.addLog("Stopping Classifier")
        self.logger.info("Stopping Classifier")

    def addLog(self, msg):
        self.message_box.addItem(msg)
        self.message_box.scrollToBottom()
