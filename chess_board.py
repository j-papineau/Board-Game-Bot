import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.image as image
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO
from roboflow import Roboflow
from shapely.geometry import Polygon
import webbrowser

def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect            
                    
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped     

def draw_grid(image, pts):
    # generate lines for FEN classification from board calibration image
    # pts is tr, tl, br, bl (x,y)
    # shape = (width, height, channel)
    
    corners = np.array([[0,0], 
                    [image.shape[1], 0], 
                    [0, image.shape[0]], 
                    [image.shape[1], image.shape[0]]])
        
    img = np.copy(image)
    
    # img = cv2.line(img, corners[0], corners[1], (0,255,0), 3)
    
    corners = order_points(corners)

    fig = figure(figsize=(10, 10), dpi=80)

    # im = plt.imread(image)
    implot = plt.imshow(image)
    
    TL = corners[0]
    BL = corners[3]
    TR = corners[1]
    BR = corners[2]

    def interpolate(xy0, xy1):
        x0,y0 = xy0
        x1,y1 = xy1
        dx = (x1-x0) / 8
        dy = (y1-y0) / 8
        pts = [(x0+i*dx,y0+i*dy) for i in range(9)]
        return pts

    ptsT = interpolate( TL, TR )
    ptsL = interpolate( TL, BL )
    ptsR = interpolate( TR, BR )
    ptsB = interpolate( BL, BR )
        
    for a,b in zip(ptsL, ptsR):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
    for a,b in zip(ptsT, ptsB):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
        
    plt.axis('off')
    
    
    # plt.savefig("./generated/chessboard_transformed_with_grid.jpg", bbox_inches='tight')
    # return img
    return ptsT, ptsL


def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

def test_new_model(img):
    
    # alpha = 2 # (1-3)
    # beta = 10 # (0-100)
    
    # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    model_trained = YOLO("models/pieces_v4.pt")
    
    results = model_trained.predict(source=img, conf=0.20, save=True)

class Chess_Board():
    def __init__(self, ptsT, ptsL, initial_img):
        self.board = []
        # self.pts = pts
        self.ptsT = ptsT 
        self.ptsL = ptsL
        self.initial_img = initial_img
        
        # init model
        print("initializing model")
        # gui.message.setText("System: Initializing Model...")
        
        
        self.model = YOLO("models/pieces_v4.pt")
        
        print("model initialized")
        
    def chess_pieces_detector(self, img):
       results = self.model.predict(source=img, conf=.30, augment=False, save_txt=True, save=True)
       
       boxes = results[0].boxes
       detections = boxes.xyxy.numpy()
       
       return detections, boxes
   
    def update_board_pos(self, img):
        print("updating board pos...")
        # img = increase_brightness(img, value=40)
        # maybe increase contrast idk
        # alpha = 3 # contrast
        # beta = 10 # brightness
        # image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        detections, boxes = self.chess_pieces_detector(img)
        print("getting detections...")
        
        xA = self.ptsT[0][0]
        xB = self.ptsT[1][0]
        xC = self.ptsT[2][0]
        xD = self.ptsT[3][0]
        xE = self.ptsT[4][0]
        xF = self.ptsT[5][0]
        xG = self.ptsT[6][0]
        xH = self.ptsT[7][0]
        xI = self.ptsT[8][0]

        y9 = self.ptsL[0][1]
        y8 = self.ptsL[1][1] 
        y7 = self.ptsL[2][1] 
        y6 = self.ptsL[3][1]  
        y5 = self.ptsL[4][1]  
        y4 = self.ptsL[5][1] 
        y3 = self.ptsL[6][1]  
        y2 = self.ptsL[7][1] 
        y1 = self.ptsL[8][1] 
        
        a8 = np.array([[xA,y9], [xB, y9], [xB, y8], [xA, y8]])
        a7 = np.array([[xA,y8], [xB, y8], [xB, y7], [xA, y7]])
        a6 = np.array([[xA,y7], [xB, y7], [xB, y6], [xA, y6]])
        a5 = np.array([[xA,y6], [xB, y6], [xB, y5], [xA, y5]])
        a4 = np.array([[xA,y5], [xB, y5], [xB, y4], [xA, y4]])
        a3 = np.array([[xA,y4], [xB, y4], [xB, y3], [xA, y3]])
        a2 = np.array([[xA,y3], [xB, y3], [xB, y2], [xA, y2]])
        a1 = np.array([[xA,y2], [xB, y2], [xB, y1], [xA, y1]])

        b8 = np.array([[xB,y9], [xC, y9], [xC, y8], [xB, y8]])
        b7 = np.array([[xB,y8], [xC, y8], [xC, y7], [xB, y7]])
        b6 = np.array([[xB,y7], [xC, y7], [xC, y6], [xB, y6]])
        b5 = np.array([[xB,y6], [xC, y6], [xC, y5], [xB, y5]])
        b4 = np.array([[xB,y5], [xC, y5], [xC, y4], [xB, y4]])
        b3 = np.array([[xB,y4], [xC, y4], [xC, y3], [xB, y3]])
        b2 = np.array([[xB,y3], [xC, y3], [xC, y2], [xB, y2]])
        b1 = np.array([[xB,y2], [xC, y2], [xC, y1], [xB, y1]])

        c8 = np.array([[xC,y9], [xD, y9], [xD, y8], [xC, y8]])
        c7 = np.array([[xC,y8], [xD, y8], [xD, y7], [xC, y7]])
        c6 = np.array([[xC,y7], [xD, y7], [xD, y6], [xC, y6]])
        c5 = np.array([[xC,y6], [xD, y6], [xD, y5], [xC, y5]])
        c4 = np.array([[xC,y5], [xD, y5], [xD, y4], [xC, y4]])
        c3 = np.array([[xC,y4], [xD, y4], [xD, y3], [xC, y3]])
        c2 = np.array([[xC,y3], [xD, y3], [xD, y2], [xC, y2]])
        c1 = np.array([[xC,y2], [xD, y2], [xD, y1], [xC, y1]])

        d8 = np.array([[xD,y9], [xE, y9], [xE, y8], [xD, y8]])
        d7 = np.array([[xD,y8], [xE, y8], [xE, y7], [xD, y7]])
        d6 = np.array([[xD,y7], [xE, y7], [xE, y6], [xD, y6]])
        d5 = np.array([[xD,y6], [xE, y6], [xE, y5], [xD, y5]])
        d4 = np.array([[xD,y5], [xE, y5], [xE, y4], [xD, y4]])
        d3 = np.array([[xD,y4], [xE, y4], [xE, y3], [xD, y3]])
        d2 = np.array([[xD,y3], [xE, y3], [xE, y2], [xD, y2]])
        d1 = np.array([[xD,y2], [xE, y2], [xE, y1], [xD, y1]])

        e8 = np.array([[xE,y9], [xF, y9], [xF, y8], [xE, y8]])
        e7 = np.array([[xE,y8], [xF, y8], [xF, y7], [xE, y7]])
        e6 = np.array([[xE,y7], [xF, y7], [xF, y6], [xE, y6]])
        e5 = np.array([[xE,y6], [xF, y6], [xF, y5], [xE, y5]])
        e4 = np.array([[xE,y5], [xF, y5], [xF, y4], [xE, y4]])
        e3 = np.array([[xE,y4], [xF, y4], [xF, y3], [xE, y3]])
        e2 = np.array([[xE,y3], [xF, y3], [xF, y2], [xE, y2]])
        e1 = np.array([[xE,y2], [xF, y2], [xF, y1], [xE, y1]])

        f8 = np.array([[xF,y9], [xG, y9], [xG, y8], [xF, y8]])
        f7 = np.array([[xF,y8], [xG, y8], [xG, y7], [xF, y7]])
        f6 = np.array([[xF,y7], [xG, y7], [xG, y6], [xF, y6]])
        f5 = np.array([[xF,y6], [xG, y6], [xG, y5], [xF, y5]])
        f4 = np.array([[xF,y5], [xG, y5], [xG, y4], [xF, y4]])
        f3 = np.array([[xF,y4], [xG, y4], [xG, y3], [xF, y3]])
        f2 = np.array([[xF,y3], [xG, y3], [xG, y2], [xF, y2]])
        f1 = np.array([[xF,y2], [xG, y2], [xG, y1], [xF, y1]])

        g8 = np.array([[xG,y9], [xH, y9], [xH, y8], [xG, y8]])
        g7 = np.array([[xG,y8], [xH, y8], [xH, y7], [xG, y7]])
        g6 = np.array([[xG,y7], [xH, y7], [xH, y6], [xG, y6]])
        g5 = np.array([[xG,y6], [xH, y6], [xH, y5], [xG, y5]])
        g4 = np.array([[xG,y5], [xH, y5], [xH, y4], [xG, y4]])
        g3 = np.array([[xG,y4], [xH, y4], [xH, y3], [xG, y3]])
        g2 = np.array([[xG,y3], [xH, y3], [xH, y2], [xG, y2]])
        g1 = np.array([[xG,y2], [xH, y2], [xH, y1], [xG, y1]])

        h8 = np.array([[xH,y9], [xI, y9], [xI, y8], [xH, y8]])
        h7 = np.array([[xH,y8], [xI, y8], [xI, y7], [xH, y7]])
        h6 = np.array([[xH,y7], [xI, y7], [xI, y6], [xH, y6]])
        h5 = np.array([[xH,y6], [xI, y6], [xI, y5], [xH, y5]])
        h4 = np.array([[xH,y5], [xI, y5], [xI, y4], [xH, y4]])
        h3 = np.array([[xH,y4], [xI, y4], [xI, y3], [xH, y3]])
        h2 = np.array([[xH,y3], [xI, y3], [xI, y2], [xH, y2]])
        h1 = np.array([[xH,y2], [xI, y2], [xI, y1], [xH, y1]])

        # transforms the squares to write FEN

        FEN_annotation = [[a8, b8, c8, d8, e8, f8, g8, h8],
                        [a7, b7, c7, d7, e7, f7, g7, h7],
                        [a6, b6, c6, d6, e6, f6, g6, h6],
                        [a5, b5, c5, d5, e5, f5, g5, h5],
                        [a4, b4, c4, d4, e4, f4, g4, h4],
                        [a3, b3, c3, d3, e3, f3, g3, h3],
                        [a2, b2, c2, d2, e2, f2, g2, h2],
                        [a1, b1, c1, d1, e1, f1, g1, h1]]
        
        board_FEN = []
        corrected_FEN = []
        complete_board_FEN = []
        
        for line in FEN_annotation:
            line_to_FEN = []
            for square in line:
                try:
                    piece_on_square = self.connect_square_to_detection(detections, square , boxes)
                    line_to_FEN.append(piece_on_square) 
                except Exception as e:
                    print("error getting piece on square")
                    print(e)
            corrected_FEN = [i.replace('empty', '1') for i in line_to_FEN]
            print(corrected_FEN)
            board_FEN.append(corrected_FEN)
            
        complete_board_FEN = [''.join(line) for line in board_FEN]
        
        to_FEN = '/'.join(complete_board_FEN)
        print("https://lichess.org/analysis/"+ to_FEN)
        webbrowser.open("https://lichess.org/analysis/"+ to_FEN)
        
        return to_FEN

    def piece_to_square():
        # names: ['black bishop', 'black king', 'black knight', 'black pawn', 'black queen', 'black rook', 'white bishop', 'white king', 'white knight', 'white pawn', 'white queen', 'white rook']
        pass  
    
    def calculate_iou(self, box_1, box_2):
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou

    def connect_square_to_detection(self, detections, square, boxes):
        
        print(f'connecting square: {square}')
    
        di = { 0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r', 6: 'B', 7: 'K', 8: 'N', 9: 'P', 10: 'Q', 11: 'R' }

        list_of_iou= []
        
        for i in detections:

            box_x1 = i[0]
            box_y1 = i[1]

            box_x2 = i[2]
            box_y2 = i[1]

            box_x3 = i[2]
            box_y3 = i[3]

            box_x4 = i[0]
            box_y4 = i[3]
            
            #cut high pieces        
            if box_y4 - box_y1 > 60:
                box_complete = np.array([[box_x1,box_y1+40], [box_x2, box_y2+40], [box_x3, box_y3], [box_x4, box_y4]])
            else:
                box_complete = np.array([[box_x1,box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])
                
            #until here

            list_of_iou.append(self.calculate_iou(box_complete, square))

        num = list_of_iou.index(max(list_of_iou))

        piece = boxes.cls[num].tolist()
        
        if max(list_of_iou) > 0.15:
            piece = boxes.cls[num].tolist()
            print(f'piece found! {di[piece]}')
            return di[piece]
        
        else:
            piece = "empty"
            return piece

def increase_brightness(img, value=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img        