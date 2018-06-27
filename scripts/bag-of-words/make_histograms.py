#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Create histogram of visual words for imageset 
"""

import numpy as np
import cv2
import argparse
import yaml
import os
import sys
import glob
import time
from scipy.spatial import distance
from random import shuffle
from joblib import Parallel, delayed


# so that relative path to archv can be used
sys.path.append(os.path.abspath(os.path.join('../..')))

from archv.classes.image import Image
from archv.classes.matches import Matches



def parse_arguments():
    """ Simple parser for command line arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument("-k", required=True, help="path to keypoint files", type=str)
    parser.add_argument("-n", default=1, help="number of cores", type=int)
    parser.add_argument("-o", required=True, help="path to histogram directory", type=str)
    parser.add_argument("-d", required=True, help="path to dictionary file", type=str)

    args = parser.parse_args()
    return args

def find_best_match(descriptor, vocab):
    min = distance.cosine(descriptor, vocab[0])
    min_index = 0
    for i,w in enumerate(vocab):
        dist = distance.cosine(descriptor, w)
        if dist < min:
            min = dist
            min_index = i
    return min_index

def process_files(fname, vocab):
    # read in keypoints
    print (fname)
    img = Image(None)
    img.read_from_file(fname)

    # histogram
    hist = []
    res = ""

    # for each descriptor, find vocab word that best matches it
    if not img.descriptors is None:
        for d in img.descriptors:
            index = find_best_match(d, vocab)
            hist.append(str(index))
        res = ",".join(hist)


    return fname, res

    

def main(args):

    # get filelist of keypoint files
    files = glob.glob(os.path.join(args.k, "*.yml"))
    filenames = []

    # filter filenames
    for fname in files:
       short = fname.split('/')[-1]
       if not os.path.isfile(args.o + short):
       	  filenames.append(fname)
       

    # read in dictionary
    dict_file = cv2.FileStorage(args.d, cv2.FILE_STORAGE_READ)
    vocab = dict_file.getNode("vocabulary").mat()

    # if doesn't exist, create output directory
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    fnames, results = Parallel(n_jobs=args.n)(delayed(process_files)(fname, vocab) for fname in filenames)
    
    # for each histogram
    for res in results:
        # write histogram to file
        ofile = args.o + fname.split('/')[-1]
        o = open (ofile, "w")
        print (res, file=o)
        print(fname, ofile)

if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    main(args)
    print ("Time Elapsed: ", time.time() - start)



        
