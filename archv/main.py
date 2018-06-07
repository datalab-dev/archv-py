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

    parser.add_argument("-i", required=True, help="path to input image directory", type=str)
    parser.add_argument("-k", required=True, help="path to keypoints for image directory", type=str)
    parser.add_argument("-p", required=True, help="path to parameter file for surf", type=str)
    parser.add_argument("-o", default="results.csv", help="path to outputfile", type=str)
    parser.add_argument("-n", default=1, help="number of cores", type=int)

    args = parser.parse_args()
    return args

def find_matches(fname, params):

    img1 = Image(fname)
    img1.compute_and_filter(params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])

    scores = []
    names = []

    kyptfiles = glob.glob(os.path.join(args.k, "*.yml"))
    for keys in kyptfiles:
        img2 = Image(None)
        img2.read_from_file(keys)

        if (len(img2.keypoints) < 2):
            continue

        matcher = Matches(img1, img2)
        matcher.ratio_test()
        matcher.symmetry_test()
        matcher.ransac_test()

        score = len(matcher.ransac_matches)

        if score > 1:
            names.append(keys.split('/')[-1].split('.')[0])
            scores.append(score)

    scores = np.array(scores)
    names = np.array(names)
    indexs = scores.argsort()
    names = names[indexs]
    scores = scores[indexs]
    matches = []
    for i in range(len(names)):
        matches.insert(0, names[i] + " " + str(scores[i]))
    results = ",".join(matches)

    return results

def main(args):
    params = yaml.load(open(args.p))
    filenames = glob.glob(os.path.join(args.i, "*.jpg"))

    full = []
    ofile = open(args.o, "w")

    full = Parallel(n_jobs=args.n)(delayed(find_matches)(fname, params) for fname in filenames)

    print (full, file=ofile)



if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    main(args)
    print ("Time Elapsed: ", time.time() - start)



        
