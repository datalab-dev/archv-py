#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import argparse
import yaml
import os
import glob
import time
from joblib import Parallel, delayed
from classes.image import Image
from classes.matches import Matches



def parse_arguments():
    """ Simple parser for command line arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="path to image", type=str)
    parser.add_argument("-k", required=True, help="path to keypoint files", type=str)
    parser.add_argument("-p", required=True, help="path to parameter file for surf", type=str)

    args = parser.parse_args()
    return args

def main(args):
    scores = []
    names = []
    results = []

    img1 = Image(args.image)
    params = yaml.load(open(args.p))
    img1.compute_and_filter(params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])

    filenames = glob.glob(os.path.join(args.k, "*.yml"))
    for fname in filenames:
        print (fname)
        img2 = Image(None)
        img2.read_from_file(fname)

        if (len(img2.keypoints) < 2):
            continue

        matcher = Matches(img1, img2)
        matcher.ratio_test()
        matcher.symmetry_test()
        matcher.ransac_test()

        score = len(matcher.ransac_matches)

        if score > 1:
            names.append(fname.split('/')[-1].split('.')[0])
            scores.append(score)

    #sort the results
    scores = np.array(scores)
    names = np.array(names)
    indexs = scores.argsort()
    names = names[indexs]
    scores = scores[indexs]
    matches = []
    for i in range(len(names)):
        matches.insert(0, names[i] + " " + str(scores[i]))
    results = ",".join(matches)
    print (results)

if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    results = main(args)
    print ("Time Elapsed: ", time.time() - start)



        
