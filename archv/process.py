#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import argparse
import yaml
import os
import glob
import time
from classes.image import Image


def parse_arguments():
    """ Simple parser for command line arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True, help="path to image directory", type=str)
    parser.add_argument("-o", required=True, help="path to output directory to store keypoint files", type=str)
    parser.add_argument("-p", required=True, help="path to parameter file for surf", type=str)

    args = parser.parse_args()
    return args

def main(args):

    # get list of all images in imagedirectory
    filenames = glob.glob(os.path.join(args.i, "*.jpg"))

    # if doesn't exist, create output directory
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # read SURF parameters into dictionary
    params = yaml.load(open(args.p))

    ## loop over all the images
    for i,fname in enumerate(filenames):

        # to track status
        if i % 50 == 0:
            print ("processing image ", i, "out of ", len(filenames) - 1)

        # generate output filename
        ofile = args.o + "/" + fname.split('/')[-1].split('.')[0] + '.yml'
        
        img = Image(fname) 
        img.compute_and_filter(params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])
        img.write_to_file(ofile)


if __name__ == "__main__":
    start = time.time()    
    args = parse_arguments()
    main(args)
    print ("Elapsed time: ", time.time() - start)


