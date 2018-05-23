#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

class Image():

    def __init__(self, image):
        self.image = None
        if image != None:
            self.image = cv2.imread(image, 0)
        self.keypoints = []
        self.descriptors = []

        # params for the keypoints
        self.minh = 0 
        self.octaves = 0
        self.layers = 0
        self.mins = 0 
        self.minr = 0.0 

    def filter_keypoints(self, min_size, min_response):
        k2 = []
        for k in self.keypoints:
            if k.size > min_size and k.response > min_response:
                k2.append(k)
        self.keypoints = k2
        self.mins = min_size
        self.minr = min_response
        return self.keypoints

    def compute_and_filter (self, minh, octaves, layers, msize, mresponse):
        self.minh = minh
        self.octaves = octaves
        self.layers = layers
        surf = cv2.xfeatures2d.SURF_create(minh, octaves, layers) #Opencv 3+
        self.keypoints = surf.detect(self.image, None)
        self.filter_keypoints(msize, mresponse)
        self.keypoints, self.descriptors = surf.compute (self.image, self.keypoints)
        return surf.compute(self.image, self.keypoints)

    def write_to_file(self, ofile):
        kps = []
        for p in self.keypoints:
            row = (p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id)
            kps.append(row)
        kps = np.array(kps)

        cv_file = cv2.FileStorage(ofile, cv2.FILE_STORAGE_WRITE)
        cv_file.write("min_hessian", self.minh)
        cv_file.write("octaves", self.octaves)
        cv_file.write("layers", self.layers)
        cv_file.write("min_size", self.mins)
        cv_file.write("min_response", self.minr)
        cv_file.write("keypoints", kps)
        cv_file.write("descriptors", self.descriptors)
        cv_file.release()
        return

    def read_from_file(self, ifile):
        kps = np.array([])
        descriptors = np.array([])
        cv_file = cv2.FileStorage(ifile, cv2.FILE_STORAGE_READ)
        kps = cv_file.getNode("keypoints").mat()
        self.descriptors = cv_file.getNode("descriptors").mat()
        self.minh = cv_file.getNode("min_hessian")
        self.octaves = cv_file.getNode("octaves")
        self.layers = cv_file.getNode("layers")
        self.mins = cv_file.getNode("min_size")
        self.minr = cv_file.getNode("min_response")
        cv_file.release()

        for p in kps:
            point = cv2.KeyPoint(x=int(p[0]), y=int(p[1]), _size=int(p[2]), _angle=int(p[3]), _response=int(p[4]), _octave=int(p[5]), _class_id=int(p[6]))
            self.keypoints.append(point)
        return self.keypoints, self.descriptors
