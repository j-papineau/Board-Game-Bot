import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def recognize_board(img_captured):
    
    # gray = cv2.cvtColor(img_captured, cv2.GRAYSCALE)
    
    GRID = (7,7)

    found, corners = cv2.findChessboardCorners(
        img_captured, GRID, cv2.CALIB_CB_ADAPTIVE_THRESH)
    
    if found:
        cv2.drawChessboardCorners(img_captured, GRID, corners, found)
        return img_captured
    else:
        print("no chessboard found")
        return False
        
    
        
    
     
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

if __name__ == '__main__':
    file = "./tests/emptyboard.jpg"
    
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while True:
        ret, frame = camera.read()
        
        if ret:
        #   cv2.imshow('camera', frame)
            tracked = recognize_board(frame)
            cv2.imshow("board", tracked)
        if cv2.waitKey(1) == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()
          
    

    
    
    # cv2.imshow("board", image)
    # cv2.waitKey(0)
    
    # cv2.destroyAllWindows()