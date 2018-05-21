import numpy as np
import cv2
import argparse
from filter import filter_keypoints
from utils import compute_and_filter


## configure, to be read in as command line arguments:
minh = 5000
octaves = 8
layers = 8
msize = 50
mresponse = 500 
image = "./images/examples/test.jpg" 



def main():

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    keypoints, descriptors = compute_and_filter (img, minh, octaves, layers, msize, mresponse)


    img = cv2.drawKeypoints(img, keypoints, None, (255,0,0),4)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

