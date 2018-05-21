#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def ratio_test(matches):
    good_matches = []
    r = .75
    for m,n in matches:
        if m and n:
            if m.distance < r*n.distance:
                good_matches.append(m)
    return good_matches

def symmetry_test(matches, matches2):
    sym_matches = []
    for m in matches:
        for t in matches2:
            if m.queryIdx == t.trainIdx and t.queryIdx == m.trainIdx:
                sym_matches.append(m)
                break
    return sym_matches

def ransac_test(keypoints, keypoints2, sym_matches):
    ransac_matches = []

    ## convert the good keypoints to point2f for cv2.findFundamentalMat
    points = []
    points2 = []
    for i in range(0, len(sym_matches)):
        index = sym_matches[i].queryIdx
        index2 = sym_matches[i].trainIdx

        point = keypoints[index].pt
        point2 = keypoints2[index2].pt

        points.append(point)
        points2.append(point2)

    pts = np.array(points)
    pts2 = np.array(points2)

    # get inliers matrix
    # need minimum of 4 points
    if len(pts) > 4 and len(pts2) > 4:
        fundamental, inliers = cv2.findFundamentalMat(pts,pts2,cv2.RANSAC)
        # use inliers to extract only the matches that are inliers
        for i, inlier in enumerate(inliers):
            if inlier != 0:
                ransac_matches.append(sym_matches[i])
    else:
        ransac_matches = sym_matches

    return ransac_matches
