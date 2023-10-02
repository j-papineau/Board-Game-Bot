from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap, QColor, QImage
from PyQt6 import uic
import os
import cv2
import numpy as np
import re
import mediapipe as mp
from sys import platform


# from ..hand_tracking import *


class Main_Tab(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QGridLayout()
        
        # state vars
        
        

        self.raw_layout = QFormLayout()
        self.FeedLabel = QLabel()
        self.raw_layout.addWidget(QLabel("raw camera"))
        self.raw_layout.addWidget(self.FeedLabel)
        self.layout.addLayout(self.raw_layout, 0, 0)
        
        self.calibrate_button = QPushButton("Calibrate Empty Board")
        self.calibrate_button.clicked.connect(self.calibrate_empty_board)
        self.raw_layout.addWidget(self.calibrate_button)

        # buttons
        self.cancel_btn = QPushButton("Stop Camera")
        self.cancel_btn.clicked.connect(self.cancel_feed)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_feed)

        # sidebar for options

        self.sidebar_layout = QFormLayout()



        # bottom buttons

        self.layout.addWidget(self.start_btn, 1, 0)
        self.layout.addWidget(self.cancel_btn, 1, 1)
        self.layout.addLayout(self.sidebar_layout, 0, 1)

        self.camera = Camera_Thread()
        self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
        self.camera.RawUpdate.connect(self.RawUpdateSlot)
        self.camera.start()

        # bottom layout for output
        self.bottom_layout = QGridLayout()


        self.layout.addLayout(self.bottom_layout, 2, 0)

        self.setLayout(self.layout)

    def ImageUpdateSlot(self, img):
        self.FeedLabel.setPixmap(QPixmap.fromImage(img))
        
    def RawUpdateSlot(self, img):
        self.raw_img = img

    def cancel_feed(self):
        self.camera.stop()

    def start_feed(self):
        self.camera.start()
        
    def calibrate_empty_boards(self):
        img = np.copy(self.raw_img)
        
        GRID = (7,7)
        
        found, corners = cv2.findChessboardCorners(img, GRID, cv2.CALIB_CB_ADAPTIVE_THRESH)
        
        if found:
            cv2.drawChessboardCorners(img, GRID, corners, found)
            
            n=0
            for i in range(0,49):
                img=cv2.putText(img,str(n), (int(corners[i,0,0]),int(corners[i,0,1])), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1, color=(0,0,255))
                n+=1
                
            # top left = 0
            # top right = 6
            # bottom left = 42
            # bottom right = 48
            
            
            tl = [int(corners[0,0,0]), int(corners[0,0,1])]
            tr = [int(corners[6,0,0]), int(corners[6,0,1])]
            bl = [int(corners[42,0,0]), int(corners[42,0,1])]
            br = [int(corners[48,0,0]), int(corners[48,0,1])]
            
            dist = abs(int(corners[0,0,0]) - int(corners[1,0,0])) + 10
            print(dist)
            
            img = img[(tl[1] - dist):(bl[1] + dist), (bl[0] - dist):(br[0] + dist)]
            
            
            # get distance between squares (approx)
            
            cv2.imshow("pee pee calibrated poo poo", img)
            cv2.waitKey(0)
        else:
            print("no board found")
        
    def calibrate_empty_boarsd(self):
        img = np.copy(self.raw_img)
        # small_frame = self.rescale_frame(img)
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        corners= cv2.goodFeaturesToTrack(gray, 100, 0.01, 50)

        for corner in corners:
            x,y= corner[0]
            x= int(x)
            y= int(y)
            cv2.rectangle(img, (x-10,y-10),(x+10,y+10),(255,0,0),-1)



        cv2.imshow("goodFeaturesToTrack Corner Detection", img)
        cv2.waitKey()
        cv2.destroyAllWindows()    

    def calibrate_empty_board(self):
       pass
    
    
class Camera_Thread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    RawUpdate = pyqtSignal(np.ndarray)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self.ThreadActive:
            ret, frame = cap.read()
            if ret:
                self.RawUpdate.emit(frame)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.flip(image, 1)
                
                qt_format = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
                pic = qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()
