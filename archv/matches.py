import numpy as np
import cv2

from archv.image import Image

def archv_matches(img1, img2):
    """ 
    Compute the matches between two images according to Archv method:
    1. Ratio test
    2. Symmetry Test
    3. Ransac Test
    """

    bf = cv2.BFMatcher()
    # compute matches, k=2 so that it computes the two closest for each keypoint (necessary for ratio test)
    m1 = bf.knnMatch(img1.descriptors, img2.descriptors, k=2) # matches 1 on 2
    m2 = bf.knnMatch(img2.descriptors, img1.descriptors, k=2) # matches 2 on 1

    # Keep only matches that pass ratio test
    good_matches1 = ratio_test(m1, r=0.75)
    good_matches2 = ratio_test(m2, r=0.75)

    # Keep only symmetric matches
    sym_matches = symmetry_test(good_matches1, good_matches2)

    # Keep only ransac matches
    matches = ransac_test(sym_matches, img1.keypoints, img2.keypoints)
    return matches

# impath1 and impath2 need to contain the images, not just the yml descriptors keypoints files!
def draw_archv_matches(impath1, impath2, ofile, params= {
        'minh':2000, 'minr':500.0, 'mins':75, 'layers':8}):
    img1 = Image(impath1, params=params)
    img2 = Image(impath2, params=params)

    matches = archv_matches(img1, img2)
    matches = [[m] for m in matches]
    img3 = cv2.drawMatchesKnn(img1.image,img1.keypoints,img2.image,img2.keypoints, matches, None ,flags=2, matchColor=(255,0,0))
    cv2.imwrite(ofile,img3)
    return

def ratio_test(matches, r=0.75):
    good_matches = []
    for m,n in matches:
        if m and n:
            if m.distance < r*n.distance:
                good_matches.append(m)
    return good_matches


def symmetry_test(matches1, matches2):
    sym_matches = []
    for m in matches1:
        for t in matches2:
            if m.queryIdx == t.trainIdx and t.queryIdx == m.trainIdx:
                sym_matches.append(m)
                break
    return sym_matches

def ransac_test(sym_matches, kp1, kp2):
    ## convert the good keypoints to point2f for cv2.findFundamentalMat
    points = []
    points2 = []
    for i in range(0, len(sym_matches)):
        index = sym_matches[i].queryIdx
        index2 = sym_matches[i].trainIdx

        point = kp1[index].pt
        point2 = kp2[index2].pt

        points.append(point)
        points2.append(point2)

    pts = np.array(points)
    pts2 = np.array(points2)

    ransac_matches = []
    # get inliers matrix
    # need minimum of 7 points
    if len(pts) > 7 and len(pts2) > 7:
        fundamental, inliers = cv2.findFundamentalMat(pts,pts2,cv2.RANSAC)
        # use inliers to extract only the matches that are inliers
        for i, inlier in enumerate(inliers):
            if inlier != 0:
                ransac_matches.append(sym_matches[i])
    return ransac_matches
