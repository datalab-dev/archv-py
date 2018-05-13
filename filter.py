#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2


# ===================================================================
#   Size and Response Time filter
# ===================================================================
def filter_keypoints(keypoints, min_size, min_response):
    filtered = []
    for k in keypoints:
        if k.size > min_size and k.response > min_response:
            filtered.append(k)
    return filtered


# ===================================================================
#   Ratio Test
#
#   taken from opencv3.4 documentation on feature matching in python
#   https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
#
#   only keep the matches where the best match is significantly
#   different from the second best match. In this case, the best
#   match is less than 80% of the second best match. The lower 
#   ratio (r) the more matches will be filtered out
#   
#   input: matches list (the output of bf.knnMatch with k=2)
#   return: matches list (all the matches that pass the ratio test)
# ===================================================================
def ratio_test(matches):
    good_matches = []
    r = .75
    for m,n in matches:
        if m.distance < r*n.distance:
            good_matches.append(m)
    return good_matches


# ===================================================================
#   Symmetry Test
# ===================================================================
def symmetry_test(matches, matches2):
    sym_matches = []
    for m in matches:
        for t in matches2:
            if m.queryIdx == t.trainIdx and t.queryIdx == m.trainIdx:
                sym_matches.append(m)
                break
    return sym_matches


