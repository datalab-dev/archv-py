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
import glob
import time
from scipy.spatial import distance
from random import shuffle
from joblib import Parallel, delayed
from classes.image import Image
from classes.matches import Matches



def parse_arguments():
    """ Simple parser for command line arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument("-k", required=True, help="path to keypoint files", type=str)
    parser.add_argument("-n", default=1, help="number of cores", type=int)
    parser.add_argument("-o", required=True, help="path to histogram directory", type=str)
    parser.add_argument("-d", required=True, help="path to dictionary file", type=str)

    args = parser.parse_args()
    return args

def get_descriptors(fname):
    img = Image(None)
    img.read_from_file(fname)
    return img.descriptors

def find_best_match(descriptor, vocab):
    min = distance.cosine(descriptor, vocab[0])
    min_index = 0
    for i,w in enumerate(vocab):
        dist = distance.cosine(descriptor, w)
        if dist < min:
            min = dist
            min_index = i
    return min_index


def main(args):

    # get filelist of keypoint files
    filenames = glob.glob(os.path.join(args.k, "*.yml"))

    # read in dictionary
    dict_file = cv2.FileStorage(args.d, cv2.FILE_STORAGE_READ)
    vocab = dict_file.getNode("vocabulary").mat()

    for fname in filenames:
        # read in keypoints
        img = Image(None)
        img.read_from_file(fname)

        # histogram
        hist = []

        # if doesn't exist, create output directory
        if not os.path.exists(args.o):
            os.makedirs(args.o)

        # for each descriptor, find vocab word that best matches it
        for d in img.descriptors:
            index = find_best_match(d, vocab)
            hist.append(vocab[index])

        hist = np.array(hist)

        # write histogram to file
        ofile = args.o + fname.split('/')[-1]
        dict_file = cv2.FileStorage(ofile, cv2.FILE_STORAGE_WRITE)
        dict_file.write("descriptors", hist)

    

if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    main(args)
    print ("Time Elapsed: ", time.time() - start)



        
