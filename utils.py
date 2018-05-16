#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

## function for parsing command line arguments


## function to write keypoints and descriptors to file
# ofile should be full path
def write_to_file(ofile, keypoints, descriptors):
    kps = []
    for p in keypoints:
        row = (p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id)
        kps.append(row)
    kps = np.array(kps)

    cv_file = cv2.FileStorage(ofile, cv2.FILE_STORAGE_WRITE)
    cv_file.write("keypoints", kps)
    cv_file.write("descriptors", descriptors)
    cv_file.release()
    return

def read_from_file(ifile):
    kps = np.array([])
    descriptors = np.array([])
    cv_file = cv2.FileStorage(ifile, cv2.FILE_STORAGE_READ)
    kps = cv_file.getNode("keypoints").mat()
    descriptors = cv_file.getNode("descriptors").mat()
    cv_file.release()

    keypoints = [] #list of keypoint objects
    for p in kps:
        point = cv2.KeyPoint(x=int(p[0]), y=int(p[1]), _size=int(p[2]), _angle=int(p[3]), _response=int(p[4]), _octave=int(p[5]), _class_id=int(p[6]))
        keypoints.append(point)

    return keypoints, descriptors

