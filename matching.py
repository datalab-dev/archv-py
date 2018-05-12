import numpy as np
import cv2
from matplotlib import pyplot as plt
from filter import ratio_test, symmetry_test

img1 = cv2.imread('./images/examples/man.jpg',0)  # queryImage
img2 = cv2.imread('./images/examples/man2.jpg',0) # trainImage

# Initiate SIFT detector
surf = cv2.xfeatures2d.SURF_create(2000, 8, 8)

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
matches2 = bf.knnMatch(des2, des1, k=2)

### Apply ratio test
good = ratio_test(matches)
good2 = ratio_test(matches2)

# Apply Symmetry test
sym = symmetry_test(good, good2)
    
print ("num matches: ", len(matches))
print ("post ratio test: ", len(good))
print ("post symmetry test: ", len(sym))

# Apply Ransac test



### cv2.drawMatchesKnn expects list of lists as matches.
##img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches, None ,flags=2)
##img4 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None ,flags=2, matchColor=(255,0,0))
##img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,sym, None ,flags=2, matchColor=(255,0,0))
##
##
##f, axarr = plt.subplots(3, 1)
##axarr[0].imshow(img3)
##axarr[1].imshow(img4)
##axarr[2].imshow(img5)
##plt.show()
###plt.imshow(img3), plt.show()

