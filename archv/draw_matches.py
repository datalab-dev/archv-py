#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Usage: python match.py /path/to/image /path/to/image2 -p /path/to/param.yml 

A simple script to visualize the matches between two images
"""

import numpy as np
import cv2
import yaml
import argparse
from matplotlib import pyplot as plt
from classes.image import Image
from classes.matches import Matches


def parse_arguments ():
    """ Basic parser for the command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("image1", help="path to query image", type=str)
    parser.add_argument("image2", help="path to train image", type=str)
    parser.add_argument("-p", default="parameters/flickr.yml", help="path to yml file surf parameters", type=str)

    args = parser.parse_args()
    return args


def main(args):
    """ detect and filter keypoints for both images, then find good matches """
    img1 = Image(args.image1)
    img2 = Image(args.image2)

    # compute keypoints and descriptors for both images
    params = yaml.load(open(args.p))
    img1.compute_and_filter(params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])
    img2.compute_and_filter(params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])

    
    matcher = Matches(img1, img2)
    matcher.ratio_test()
    matcher.symmetry_test()
    matcher.ransac_test()

    r2 = []
    for r in matcher.ransac_matches:
        r2.append([r])

    print ("num matches: ", len(matcher.matches1))
    print ("post ratio test: ", len(matcher.good_matches1))
    print ("post symmetry test: ", len(matcher.symm_matches))
    print ("post ransac test: ", len(matcher.ransac_matches))

    img3 = cv2.drawMatchesKnn(img1.image,img1.keypoints,img2.image,img2.keypoints, r2, None ,flags=2, matchColor=(255,0,0))
    plt.imshow(img3), plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
