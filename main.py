
import sys
import app
import logging

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication

def main():
    ''' Creates UI and starts GUI event loop '''
    apollo = QApplication(sys.argv)
    # Start Application
    main_window = app.MainWindow()
    main_window.show()
    sys.exit(apollo.exec_())

if __name__ == "__main__":

    logging.basicConfig(
        format='''%(asctime)s [%(levelname)s] %(message)s''',
        datefmt="%H:%M:%S",
        filename="session.log", filemode='w',
        encoding="utf-8",
        level=logging.DEBUG
    )

    main()
