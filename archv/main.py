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
    """ Simple parser for yml settings file """
    params= yaml.load(open("settings.yml"))
    return params

def print_parameters(params):
    print ("images: ", params["imagedir"], " keypoints: ", params["keypointdir"], " output: ", params["outputfile"])
    print ("num_cores: ", params["ncores"], " ratio: ", params["ratio"])
    print ("SURF: ", params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])

def sort_results(names, scores):
    matches = []

    scores = np.array(scores)
    names = np.array(names)
    indexs = scores.argsort()
    names = names[indexs]
    scores = scores[indexs]
    for i in range(len(names)):
        matches.insert(0, names[i] + " " + str(scores[i]))
    return ",".join(matches)


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

        if score > 0:
            names.append(keys.split('/')[-1].split('.')[0])
            scores.append(score)

    results = sort_results(names, scores)
    return results

def main(params):
    print_parameters(params)
    filenames = glob.glob(os.path.join(params["imagedir"], "*.jpg"))

    full = []
    full = Parallel(n_jobs=params["ncores"])(delayed(find_matches)(fname, params) for fname in filenames)

    ofile = open(params["outputfile"], "w")
    print (full, file=ofile)



if __name__ == "__main__":
    start = time.time()
    params = parse_arguments()
    main(params)
    print ("Time Elapsed: ", time.time() - start)



        
