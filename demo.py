import cv2
from chess_board import *


def parse_board():
    print("parsing chess board")

    img = cv2.imread("./tests/average.jpg")

    image = np.copy(img)

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()
    # CHANGE

    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    
    (corners, ids, rejected) = detector.detectMarkers(image)

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
            pts = np.array(pts)
            calibrated = True
            warped_img = four_point_transform(image, pts)
            calibration_image = np.copy(warped_img)
            
            cv2.imshow("warped", warped_img)

            ptsT, ptsL = draw_grid(warped_img, pts)

            gridImg = cv2.imread("./generated/chessboard_transformed_with_grid.jpg")
            cv2.imshow("grid", gridImg)

            print("running chess piece detection")

            chess_pieces_detector(warped_img)
            
            

            # cv2.imshow("Corners", image)
            # cv2.imshow("Calibrated IMG", warped_img)
            # cv2.waitKey(0)
    else:
        print("ERROR: one or more corners not found") 



    cv2.imshow("raw", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parse_board()