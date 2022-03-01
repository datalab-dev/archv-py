import glob

import cv2

from archv.archv import compute_and_filter
from archv.archv import archv_match

seedfile = "./sample_data/ballads-100/A-20707-30.jpg"
seed = cv2.imread(seedfile, 0)

imagefiles = glob.glob("./sample_data/ballads-100/*.jpg")
images = [cv2.imread(i,0) for i in imagefiles]

# SIFT test
minr = 0
mins = 0
detector = cv2.SIFT_create()

# SURF test
#minr = 500
#mins = 75
#detector = cv2.xfeatures2d.SURF_create(2000, 8, 8)

for i in range(len(images)):
    m = archv_match(seed, images[i], minr, mins, detector, keep=True)
    if len(m) > 0:
        print(imagefiles[i], str(len(m)))
