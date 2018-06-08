#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Read random sample of keypoint files and cluster the keypoints to make visual dictionary
"""

import numpy as np
import cv2
import argparse
import yaml
import os
import sys
import glob
import time
from random import shuffle
from joblib import Parallel, delayed

# so that relative path to archv can be used
sys.path.append(os.path.abspath(os.path.join('../..')))

from archv.classes.image import Image
from archv.classes.matches import Matches



def parse_arguments():
    """ Simple parser for command line arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", default=1000, help="size of visual vocabulary", type=int)
    parser.add_argument("-k", required=True, help="path to keypoint files", type=str)
    parser.add_argument("-n", default=1, help="number of cores", type=int)
    parser.add_argument("-o", required=True, help="path to dictionary.yml", type=str)

    args = parser.parse_args()
    return args

def main(args):

    # get 300 keypoint files at random
    filenames = glob.glob(os.path.join(args.k, "*.yml"))
    shuffle(filenames) #random
    filenames = filenames[0:300] #sample

    bow = cv2.BOWKMeansTrainer(args.s)

    # get list of all descriptors for the sample
    dlist = []
    for fname in filenames:
        img = Image(None)
        img.read_from_file(fname)
        if not img.descriptors is None:
            dlist.append(img.descriptors)

    # load all descriptors into numpy matrix
    td = []
    for descriptors in dlist:
        for d in descriptors:
            td.append(d)
    td = np.array(td)

    # cluster the descriptors to vocab size
    bow.add(td)
    vocab = bow.cluster()

    #write to file
    dict_file = cv2.FileStorage(args.o, cv2.FILE_STORAGE_WRITE)
    dict_file.write("vocabulary", vocab)



if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    main(args)
    print ("Time Elapsed: ", time.time() - start)



        
