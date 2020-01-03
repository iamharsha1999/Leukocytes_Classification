import cv2
import numpy as np

def callback(x):
    pass

frame = cv2.imread('Dataset/Image_Classification/Neutrophils/118.jpg')
cv2.namedWindow('image')

ilowR = 0
ihighR = 255

ilowG = 0
ihighG = 255

ilowB = 0
ihighB = 255

# create trackbars for color change
cv2.createTrackbar('lowR','image',ilowR,255,callback)
cv2.createTrackbar('highR','image',ihighR,255,callback)

cv2.createTrackbar('lowG','image',ilowG,255,callback)
cv2.createTrackbar('highG','image',ihighG,255,callback)

cv2.createTrackbar('lowB','image',ilowB,255,callback)
cv2.createTrackbar('highB','image',ihighB,255,callback)



while(1):
    # get trackbar positions
    ilowR = cv2.getTrackbarPos('lowR', 'image')
    ihighR = cv2.getTrackbarPos('highR', 'image')
    ilowG = cv2.getTrackbarPos('lowG', 'image')
    ihighG = cv2.getTrackbarPos('highG', 'image')
    ilowB = cv2.getTrackbarPos('lowB', 'image')
    ihighB = cv2.getTrackbarPos('highB', 'image')
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', frame)
    lower_rgb = np.array([ilowR, ilowG, ilowB])
    higher_rgb = np.array([ihighR, ihighG, ihighB])
    mask = cv2.inRange(frame, lower_rgb, higher_rgb)
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


cv2.destroyAllWindows()
