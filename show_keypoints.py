import cv2
import numpy as np
 
img = cv2.imread("./images/examples/test.jpg", cv2.IMREAD_GRAYSCALE)

#minh, noctaves, nlayers
surf = cv2.xfeatures2d.SURF_create(5000, 8, 8) #Opencv 3+
 
keypoints, descriptors = surf.detectAndCompute(img, None)
print(len(keypoints))
 
img = cv2.drawKeypoints(img, keypoints, None, (255,0,0),4)
 
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
