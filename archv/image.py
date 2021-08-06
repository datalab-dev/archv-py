import numpy as np
import cv2

class Image():
    """ 
    A class to represent an image. Handles loading the image from file
    as an opencv image, as well as provides functions for computing
    keypoints + descriptors for the image. Can save and load keypoints
    and descriptors from yaml file.

    Attributes
    ----------
    name: str
        image filename
    keypoints: list
        list of keypoint vectors for the image
    descriptors: list
        list of descriptor vectors for the image
    params: dictionary
        A dictionary of SURF parameters. 
        Keys are minh (minimum hessian), octaves, layers, mins (minimum size), minr (minimum response)

    Methods
    -------
    filter_keypoints(mins, minr):
        filters keypoints that are smaller than mins or minr
    computer_and_filter(minh, octaves, layers, mins, minr):
       computes and filters keypoints for an image based on SURF parameters
    read_from_file(ifile):
       reads keypoints and descriptors for an image from yaml file
    write_to_file(oftile):
       writes keypoints and descriptors for an image to yaml file
    
    """

    def __init__(self, imagepath, params={
            'minh':2000, 'minr':500.0, 'mins':75, 'octaves':8, 'layers':8}):

        # if path is yml (load from yml)
        # TODO

        # else if image is jpg, png, pdf etc..
        self.image = cv2.imread(imagepath, 0)
        self.name = imagepath
        self.params = params
        self.keypoints, self.descriptors = self.compute_and_filter()


    def compute_and_filter (self):
        """ Compute SURF keypoints for the image """

        surf = cv2.xfeatures2d.SURF_create(self.params["minh"], 
                self.params["octaves"], self.params["layers"])
        self.keypoints = surf.detect(self.image, None)

        # filter keypoints
        self.keypoints = [k for k in self.keypoints 
                if k.size > self.params["minhs"] and 
                k.response > self.params["minr"]] 

        # get descriptors
        return surf.compute(self.image, self.keypoints)

    def write_to_file(self, ofile):
        """ 
            Save OpenCV generated YAML with the following fields:
                fpath,
                num_keypoints, 
                min_hessian, 
                octaves, 
                layers, 
                min_size, 
                min_response, 
                keypoints, 
                descriptors 
        """
        kps = []
        for p in self.keypoints:
            row = (p.pt[0], p.pt[1], p.size, p.angle, p.response, 
                    p.octave, p.class_id)
            kps.append(row)
        kps = np.array(kps)

        cv_file = cv2.FileStorage(ofile, cv2.FILE_STORAGE_WRITE)
        cv_file.write("num_keypoints", len(self.keypoints))
        cv_file.write("min_hessian", self.params["minh"])
        cv_file.write("octaves", self.params["octaves"])
        cv_file.write("layers", self.params["layers"])
        cv_file.write("min_size", self.params["minh"])
        cv_file.write("min_response", self.params["minr"])

        if len(self.keypoints) > 0:
            cv_file.write("keypoints", kps) 
            cv_file.write("descriptors", self.descriptors)

        cv_file.release()
        return

    def read_from_file(self, ifile):
        """
            Read YAML file created from this class 
        """
        kps = np.array([])
        descriptors = np.array([])
        cv_file = cv2.FileStorage(ifile, cv2.FILE_STORAGE_READ)
        self.params["minh"] = cv_file.getNode("min_hessian")
        self.params["octaves"] = cv_file.getNode("octaves")
        self.params["layers"] = cv_file.getNode("layers")
        self.params["mins"] = cv_file.getNode("min_size")
        self.params["minr"] = cv_file.getNode("min_response")

        kps = cv_file.getNode("keypoints").mat()
        self.descriptors = cv_file.getNode("descriptors").mat()
        cv_file.release()

        if not kps is None:
            for p in kps:
                point = cv2.KeyPoint(x=int(p[0]), y=int(p[1]), _size=int(p[2]),
                        _angle=int(p[3]), _response=int(p[4]), 
                        _octave=int(p[5]), _class_id=int(p[6]))
                self.keypoints.append(point)

        return self.keypoints, self.descriptors
