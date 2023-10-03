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
from chess_board import *
import chess
from chessboard import display



# from ..hand_tracking import *


class Main_Tab(QtWidgets.QWidget):
    
    ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QGridLayout()
        
        # state vars
        
        
        
        self.calibrated = False
        self.counter = 0

        self.raw_layout = QFormLayout()
        self.FeedLabel = QLabel()
        self.raw_layout.addWidget(QLabel("raw camera"))
        self.raw_layout.addWidget(self.FeedLabel)
        self.layout.addLayout(self.raw_layout, 0, 0)
        
        self.calibrate_button = QPushButton("Calibrate Board")
        self.calibrate_button.setToolTip("Make sure all four corner codes are visible")
        self.calibrate_button.setMinimumHeight(50)
        self.calibrate_button.clicked.connect(self.calibrate_empty_board)
        self.raw_layout.addWidget(self.calibrate_button)
        
        self.calib_layout = QFormLayout()
        self.CalibLabel = QLabel("Calibation not run")
        self.frame_label = QLabel("Calibration Check")
        self.frame_label.setStyleSheet("font-size: 22px;")
        self.calib_layout.addWidget(self.frame_label)
        self.calib_layout.addWidget(QLabel("This is a static image (duh)"))
        self.calib_layout.addWidget(self.CalibLabel)
        self.layout.addLayout(self.calib_layout, 0, 1)
        
        self.show_board_layout = QFormLayout()
        self.BoardLabel = QLabel("Board not calibrated yet :3")
        self.show_board_label = QLabel("Board (after lots of math)")
        self.show_board_label.setStyleSheet("font-size: 22px;")
        self.show_board_layout.addWidget(self.show_board_label)
        self.show_board_layout.addWidget(self.BoardLabel)
        self.show_board_layout.setVerticalSpacing(10)
        self.layout.addLayout(self.show_board_layout, 1, 0)
        
        self.digital_board_layout = QFormLayout()
        self.board = chess.Board()
        
        
        
        
        
        
        
        
        self.confirm_calibration = QPushButton("Approve Calibration")
        self.confirm_calibration.setMinimumHeight(50)
        self.confirm_calibration.setEnabled(False)
        self.confirm_calibration.clicked.connect(self.draw_chess_board)
        self.calib_layout.addWidget(self.confirm_calibration)

        # buttons
        self.cancel_btn = QPushButton("Stop Camera")
        self.cancel_btn.clicked.connect(self.cancel_feed)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_feed)

        # sidebar for options

        self.sidebar_layout = QFormLayout()



        # bottom buttons

        self.layout.addWidget(self.start_btn, 2, 0)
        self.layout.addWidget(self.cancel_btn, 2, 1)
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
        self.rawQPic = img
        self.FeedLabel.setPixmap(QPixmap.fromImage(img))
        
    def RawUpdateSlot(self, img):
        self.raw_img = img
        self.counter += 1
        if self.counter % 2 == 0:
            self.CalibratedUpdateSlot()
        
    def CalibratedUpdateSlot(self):
        pass

    def cancel_feed(self):
        self.camera.stop()

    def start_feed(self):
        self.camera.start()    
    
    def calibrate_empty_board(self):
        image = np.copy(self.raw_img)
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        
        if len(corners) > 0:
            
            ids = ids.flatten()
            
            code_coords = []
            pts = []
            
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))  
                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[INFO] ArUco marker ID: {}".format(markerID))
                
                print(f'topleft: {topLeft}')
                
                parsed_info = {
                    "id": markerID,
                    "top-left": topLeft,
                    "top-right": topRight,
                    "bottom-left": bottomLeft,
                    "bottom-right": bottomRight,
                    "center-x": cX,
                    "center-y": cY, 
                }
                
                # WILL NEED TO CHANGE THIS FOR ACCURACY
                pts.append([cX, cY])
                
                code_coords.append(parsed_info)
                # show the output image
            
            # warp board based on corners
            # 2 is top left, 0 is top right
            # 3 is bottom left, 1 is bottom right
            topRight, topLeft, bottomRight, bottomLeft = {}, {}, {}, {}
            
            for coord in code_coords:
                if coord['id'] == 0:
                    topRight = (coord["bottom-left"])
                    break
                else:
                    topRight = None
                
            for coord in code_coords:
                if coord['id'] == 2:
                    topLeft = (coord["bottom-right"])
                    break
                else:
                    topLeft = None
            for coord in code_coords:
                if coord['id'] == 3:
                    bottomLeft = (coord["top-right"])
                    break
                else:
                    bottomLeft = None
            for coord in code_coords:
                if coord['id'] == 1:
                    bottomRight = (coord["top-left"])
                    break
                else:
                    bottomRight = None
                    
            pts = [topRight, topLeft, bottomLeft, bottomRight]
                    
            if not [x for x in (topRight, topLeft, bottomRight, bottomLeft) if x is None]:
                print("All four corners found")
                self.pts = np.array(pts)
                self.calibrated = True
                warped_img = four_point_transform(image, self.pts)
                self.calibration_image = np.copy(warped_img)
                
                # display calibration image for checking
                height, width, channel = warped_img.shape
                bytesPerLine = 3 * width
                qImg = QImage(warped_img.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
                self.CalibLabel.setPixmap(QPixmap.fromImage(qImg))
                self.confirm_calibration.setEnabled(True)
                
                

                # cv2.imshow("Corners", image)
                # cv2.imshow("Calibrated IMG", warped_img)
                # cv2.waitKey(0)
            else:
                print("ERROR: one or more corners not found") 
    
    def draw_chess_board(self):
        print("Parsing Board")
        self.ptsT, self.ptsL = draw_grid(self.calibration_image, self.pts)
        
        img = QImage("./generated/chessboard_transformed_with_grid.jpg")
        img = img.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
        self.BoardLabel.setPixmap(QPixmap.fromImage(img))
        
        # display.start(self.board.board_fen())
            
        # self.ptsT, self.ptsL = draw_grid(self.calibration_image, self.pts)
        # print(self.ptsT)                       
      
                
    
    
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
