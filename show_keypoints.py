#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Usage: python show_keypoints.py 2000 8 8 50 500 /path/to/image

A simple script that is used for tuning the surf parameters used by archv for the imageset. 
This is important becuase each imageset has generally different levels of detail and can require very different
surf parameters in order to have a good balance between many features and run time. 
"""

import numpy as np
import cv2
import argparse
from filter import filter_keypoints
from utils import compute_and_filter

def parse_arguments ():
    """ 
    Basic parser for the required positional arguments to show_keypoints 
    returns dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("minh", help="Set the threshold for minhessian",
            type=int)
    parser.add_argument("octaves", help="Set the number of octaves of scale space for the image",
            type=int)
    parser.add_argument("layers", help="Set the number of octave layers",
            type=int)
    parser.add_argument("msize", help="Set the minimum size of keypoints",
            type=int)
    parser.add_argument("mresponse", help="Set the minimum response of keypoints",
            type=float)
    parser.add_argument("image", help="path to input image file",
            type=str)
    args = parser.parse_args()
    return args

def main(args):
    """
    detect, filter and display the keypoints found for a given input image using specified SURF parameters 

    arguments are passed in from the parse_arguments function. order for arguments:
    show_keypoints.py minh octaves layers mszie mresponse path/to/image
    """ 

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    keypoints, descriptors = compute_and_filter (img, args.minh, args.octaves, args.layers, args.msize, args.mresponse)


    img = cv2.drawKeypoints(img, keypoints, None, (255,0,0),4)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

