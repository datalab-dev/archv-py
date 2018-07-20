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
    """ Simple parser for yml settings file """
    params= yaml.load(open("settings.yml"))
    return params

def print_parameters(params):
    print ("images: ", params["imagedir"], " keypoints: ", params["keypointdir"])
    print ("num_cores: ", params["ncores"], " ratio: ", params["ratio"])
    print ("SURF: ", params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])


def process_image(img, params):
    ofile = args.o + "/" + img.name.split('/')[-1].split('.')[0] + '.yml'
    img.compute_and_filter(params["min_hessian"], params["octaves"], params["layers"], params["min_size"], params["min_response"])
    img.write_to_file(ofile)

def main(params):
    print_parameters(params)

    # get list of all images in imagedirectory
    filenames = glob.glob(os.path.join(params["imagedir"], "*.jpg"))

    # if doesn't exist, create output directory
    if not os.path.exists(params["keypointdir"]):
        os.makedirs(params["keypointdir"])


    images = []
    # get a list of Images  
    for i, fname in enumerate(filenames):
        if i % 50 == 0:
           print ("initializing image class for: ", i, fname)
        image = Image(fname)
        images.append(image)


    Parallel(n_jobs=params["ncores"])(delayed(process_image)(img, params) for img in images)

        
if __name__ == "__main__":
    start = time.time()    
    params = parse_arguments()
    main(params)
    print ("Elapsed time: ", time.time() - start)


