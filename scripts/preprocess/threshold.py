import cv2
import numpy as np


image = "image.tif"
img = cv2.imread(image, 0)

equ = cv2.equalizeHist(img)
retval, threshold = cv2.threshold(equ, 100, 255, cv2.THRESH_BINARY)

res = np.hstack((threshold,equ)) #stacking images side-by-side

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', res)
cv2.waitKey(0)
cv2.destroyWindow('image')
