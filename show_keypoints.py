#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Usage: python show_keypoints.py /path/to/image -minh 2000 -o 8 -l 8 -s 50 -r 500 
   or: python show_keypoints.py /path/to/image -k /path/to/keypointfile

A simple script that is used for tuning the surf parameters used by archv for the imageset. 
This is important becuase each imageset has generally different levels of detail and can require very different
surf parameters in order to have a good balance between many features and run time. 
"""

import numpy as np
import cv2
import sys
import argparse
from filter import filter_keypoints
from utils import compute_and_filter, read_from_file

def parse_arguments ():
    """ 
    Basic parser for the required positional arguments to show_keypoints 
    returns dictionary of arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="path to input image file", type=str)

    parser.add_argument("-k", help="path to yml file containing keypoints and descriptors", type=str, default=None)

    parser.add_argument("-minh", help="Set the threshold for minhessian", type=int, default=2000)
    parser.add_argument("-o", help="Set the number of octaves of scale space for the image", type=int, default=8)
    parser.add_argument("-l", help="Set the number of octave layers", type=int, default=8)
    parser.add_argument("-s", help="Set the minimum size of keypoints", type=int, default=50)
    parser.add_argument("-r", help="Set the minimum response of keypoints", default=500.0)

    args = parser.parse_args()
    return args

def main(args):
    """
    detect, filter and display the keypoints found for a given input image using specified SURF parameters 

    arguments are passed in from the parse_arguments function. order for arguments:
    show_keypoints.py minh octaves layers mszie mresponse path/to/image
    """ 
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    if args.k:
        keypoints, descriptors = read_from_file(args.k)

    else:
        keypoints, descriptors = compute_and_filter (img, args.minh, args.o, args.l, args.s, args.r)


    img = cv2.drawKeypoints(img, keypoints, None, (255,0,0),4)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

