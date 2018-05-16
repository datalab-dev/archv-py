#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
from filter import ratio_test, symmetry_test, ransac_test, filter_keypoints
from utils import write_to_file

## read in input directory
inputdir = "/home/arthur/imageset/sample/"
outputdir = "./output/keypoints/"
filenames = glob.glob(os.path.join(inputdir, "*.jpg"))

for i,fname in enumerate(filenames):
    print ("processing image ", i, " out of ", len(filenames) - 1)
    img = cv2.imread(fname,0)
    ofile = outputdir + fname.split('/')[-1] + ".yml" 

    surf = cv2.xfeatures2d.SURF_create(3000, 10, 10)
    keys = surf.detect(img, None)
    keypoints = filter_keypoints(keys, 50, 500)
    temp, descriptors = surf.compute(img, keypoints)

    write_to_file(ofile, keypoints, descriptors)
    


