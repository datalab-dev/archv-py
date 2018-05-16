#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

## function for parsing command line arguments


## function to write keypoints and descriptors to file
# ofile should be full path
def write_to_file(ofile, keypoints, descriptors):
    kps = np.array([])
    for p in keypoints:
        row = (p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id)
        kps = np.append(kps, row)

    cv_file = cv2.FileStorage(ofile, cv2.FILE_STORAGE_WRITE)
    cv_file.write("keypoints", kps)
    cv_file.write("descriptors", descriptors)
    cv_file.release()
    return

