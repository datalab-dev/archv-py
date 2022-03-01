import glob

import cv2

from archv.archv import compute_and_filter
from archv.archv import archv_match_precomputed

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

# test saving to file

s_kp, s_desc = compute_and_filter(seed, minr, mins, detector, ofile="test.yml")

for i in range(len(images)):

    # compute keypoints 
    kp, desc = compute_and_filter(images[i], minr, mins, detector)
    m = archv_match_precomputed(s_kp, kp, s_desc, desc)

    if len(m) > 0:
        print(imagefiles[i], str(len(m)))
