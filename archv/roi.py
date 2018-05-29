#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import archv.classes

def parse_arguments ():
    """ Basic parser for the command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to input image file", type=str)
    args = parser.parse_args()
    return args

def main(args):

    img = cv2.imread(img.image) 

    r = cv2.selectROI(img)

    cropped = img.image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

