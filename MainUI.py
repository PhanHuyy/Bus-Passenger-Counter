from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic,QtCore
import sys
import cv2
from backdetect import *
from doordetect import *
import numpy as np


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()



        uic.loadUi("UI.ui",self)
        self.image = None

        self.button = self.findChild(QPushButton,"openfile")
        self.button.setGeometry(124, 48, 160, 24)
        self.button3 = self.findChild(QPushButton, "openfile_2")
        self.button3.setGeometry(1, 48,  115, 24)
        self.button2 = self.findChild(QPushButton, "run")
        self.button2.setGeometry(1700, 1000, 180, 40)
        self.button2.clicked.connect(self.close)
        self.label = self.findChild(QLabel,"label")
        self.label = self.findChild(QLabel, "label_2")
        self.combobox1 = self.findChild(QComboBox,"comboBox1")
        self.combobox2 = self.findChild(QComboBox, "comboBox2")


        self.button.clicked.connect(self.openfiles)
        self.button3.clicked.connect(self.webcam)

        # create a text label

        # create a vertical box layout and add the two labels
        # set the vbox layout as the widgets layout

    def openfiles(self):
        self.fname = QFileDialog.getOpenFileName(self,"Open File", "", "All Files (*);;Images (*)" )
        input = str(self.fname[0])
        input.encode('unicode_escape').decode()
        combo1 = str(self.combobox1.currentText())
        combo2 = str(self.combobox2.currentText())
        if combo2 == "Door Detection":
            run2(source=input,resolution=combo1)
        else:
            run(source=input,resolution=combo1)



    def webcam(self):
        'UIWindow.hide()'
        self.openwebcam()

    def openwebcam(self):
        input = 0
        combo1 = str(self.combobox1.currentText())
        combo2 = str(self.combobox2.currentText())
        if combo2 == "Door Detection":
            run2(source=input, resolution=combo1)
        else:
            run(source=input, resolution=combo1)




app = QApplication(sys.argv)
UIWindow = GUI()
UIWindow.show()
app.exec()