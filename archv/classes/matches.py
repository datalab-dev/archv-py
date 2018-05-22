#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from .image import Image

class Matches():

    def __init__(self, img1, img2):
        self.image1 = img1 
        self.image2 = img2 

        # compute matches image 1 on image 2 and image 2 on image 1
        bf = cv2.BFMatcher()
        self.matches1 = bf.knnMatch(self.image1.descriptors, self.image2.descriptors, k=2) #k=2 for ratiotest
        self.matches2 = bf.knnMatch(self.image1.descriptors, self.image2.descriptors, k=2)

        self.good_matches1 = []
        self.good_matches2 = []
        self.symm_matches = []
        self.ransac_matches = []
        return

    def ratio_test(self):
        r = .75
        for m,n in self.matches1:
            if m and n:
                if m.distance < r*n.distance:
                    self.good_matches1.append(m)
        for m,n in self.matches2:
            if m and n:
                if m.distance < r*n.distance:
                    self.good_matches2.append(m)
        return self.good_matches1, self.good_matches2


    def symmetry_test(self):
        for m in self.good_matches1:
            for t in self.good_matches2:
                if m.queryIdx == t.trainIdx and t.queryIdx == m.trainIdx:
                    self.symm_matches.append(m)
                    break
        return self.symm_matches

    def ransac_test(self):
        ## convert the good keypoints to point2f for cv2.findFundamentalMat
        points = []
        points2 = []
        for i in range(0, len(self.symm_matches)):
            index = self.symm_matches[i].queryIdx
            index2 = self.symm_matches[i].trainIdx

            point = self.image1.keypoints[index].pt
            point2 = self.image2.keypoints[index2].pt

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
                    self.ransac_matches.append(self.symm_matches[i])
        else:
            self.ransac_matches = self.symm_matches

        return self.ransac_matches
