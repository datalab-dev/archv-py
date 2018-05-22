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


def parse_arguments():
    """ Simple parser for command line arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True, help="path to image directory", type=str)
    parser.add_argument("-o", required=True, help="path to output directory to store keypoint files", type=str)
    parser.add_argument("-p", required=True, help="path to parameter file for surf", type=str)
    parser.add_argument("-n", default=1, help="number of cores", type=int)

    args = parser.parse_args()
    return args

def process_image(fname, params):
    ofile = args.o + "/" + fname.split('/')[-1].split('.')[0] + '.yml'
    print ("processing ", ofile)

    img = Image(fname) 
    img.compute_and_filter(params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])
    img.write_to_file(ofile)
    return

def main(args):

    # get list of all images in imagedirectory
    filenames = glob.glob(os.path.join(args.i, "*.jpg"))

    # if doesn't exist, create output directory
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # read SURF parameters into dictionary
    params = yaml.load(open(args.p))

    Parallel(n_jobs=args.n)(delayed(process_image)(fname, params) for fname in filenames)
        
if __name__ == "__main__":
    start = time.time()    
    args = parse_arguments()
    main(args)
    print ("Elapsed time: ", time.time() - start)


