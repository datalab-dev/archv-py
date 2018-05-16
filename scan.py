#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
from filter import ratio_test, symmetry_test, ransac_test, filter_keypoints
from utils import read_from_file


### parameters
input = "seed.jpg"
imagedir = "/home/arthur/imageset/sample/"
keypointdir = "./output/keypoints/"
filenames = glob.glob(os.path.join(keypointdir, "*.yml"))
outputfile = "matches_" + input.split('.')[0] + ".txt"
ofile = open(outputfile, "w")

### process input image
img = cv2.imread(input,0) # trainImage
surf = cv2.xfeatures2d.SURF_create(3000, 10, 10)
keys = surf.detect(img, None)
keys1 = filter_keypoints (keys, 50, 500)
temp, des1 = surf.compute(img, keys1)

### process all the keypoint files
filenames = glob.glob(os.path.join("./output/keypoints/", "*.yml"))
scores = []
for i,fname in enumerate(filenames):

    if (i % 50 == 0):
        print ("processing image ", i,  " out of ", len(filenames) - 1)

    #read in keypoints and descriptors for each image
    keys2, des2 = read_from_file (fname)

    if (len(keys2) < 2):
        continue

    # find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    matches2 = bf.knnMatch(des2, des1, k=2)
    
    # apply tests
    good = ratio_test(matches)
    good2 = ratio_test(matches2)
    sym = symmetry_test(good, good2)
    ransac = ransac_test (keys1, keys2, sym) 

    # store results
    scores.append(fname.split('/')[-1] + " " + str(len(ransac)))

for s in scores:
    print (s, file=ofile)
