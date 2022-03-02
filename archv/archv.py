import numpy as np
import cv2

def compute_and_filter(image, minr, mins, detector, ofile=None):
    kps = detector.detect(image, None)
    # filter keypoints on size and response
    kps = [k for k in kps if k.size > mins and k.response > minr]
    kps, descriptors = detector.compute(image, kps)

    if ofile:
        write_to_file(kps, descriptors, minr, mins, ofile)

    return (kps, descriptors)

def write_to_file(keys, descriptors, minr, mins, ofile):
    kps = []
    for p in keys:
        row = (p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id)
        kps.append(row)
    kps = np.array(kps)

    cv_file = cv2.FileStorage(ofile, cv2.FILE_STORAGE_WRITE)
    cv_file.write("num_keypoints", len(keys)) 
    cv_file.write("min_size", mins)
    cv_file.write("min_response", minr)

    # dont see how to get the parameters from detector 
    # will have to save the config elsewhere (maybe in odir name?)

    if len(keys) > 0:
        cv_file.write("keypoints", kps)
        cv_file.write("descriptors", descriptors)
    cv_file.release()
    return

def read_from_file(ifile):
    kps = np.array([])
    desc = np.array([])
    cv_file = cv2.FileStorage(ifile, cv2.FILE_STORAGE_READ)
    kps = cv_file.getNode("keypoints").mat()
    desc = cv_file.getNode("descriptors").mat()

    keys = []
    if not kps is None:
        for p in kps:
            point = cv2.KeyPoint(x=int(p[0]), y=int(p[1]), size=int(p[2]),
                    angle=int(p[3]), response=int(p[4]), 
                    octave=int(p[5]), class_id=int(p[6]))
            keys.append(point)

    return keys, desc

def archv_match_precomputed(kp1,kp2, desc1, desc2):
    bf = cv2.BFMatcher()
    m1 = bf.knnMatch(desc1, desc2, k=2)
    m2 = bf.knnMatch(desc2, desc1, k=2)

    # ratio test
    m1 = ratio_test(m1, r=0.75)
    m2 = ratio_test(m2, r=0.75)

    # sym test
    sym = sym_test(m1, m2)
    
    # ransac
    matches = ransac_test(sym, kp1, kp2)
    return matches


def archv_match(seed, image, minr, mins, detector, keep=False):
    kps1,desc1 = compute_and_filter(seed, minr, mins, detector)
    kps2,desc2 = compute_and_filter(image, minr, mins, detector)

    matches = archv_match_precomputed(kps1, kps2, desc1, desc2)

    if keep:
        return [matches, kps1, desc1, kps2, desc2]
    else:
        return matches

def draw_matches(seed, image, minr, mins, detector, ofile=""):
    matches, kps1, desc1, kps2, desc2 = archv_match(seed, image, minr, mins, detector, keep=True)
    matches = [[m] for m in matches]
    img = cv2.drawMatchesKnn(seed, kps1, image, kps2, matches, None, flags=2, matchColor=(255,0,0))
    cv2.imwrite(ofile, img)
    return

def ratio_test(matches, r=0.75):
    passing = []
    for m,n in matches:
        if m and n:
            if m.distance < r*n.distance:
                passing.append(m)
    return passing

def sym_test(matches1, matches2):
    sym = []
    for m in matches1:
        for t in matches2:
          if m.queryIdx == t.trainIdx and t.queryIdx == m.trainIdx:
              sym.append(m)
              break
    return sym

def ransac_test(matches, kp1, kp2):
    ransac_matches = []

    # get points from matches for cv2.findFundamentalMat
    points1 = np.array([kp1[m.queryIdx].pt for m in matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in matches])

    if len(points1) < 7 or len(points2) < 7:
        return []

    fun,inliers = cv2.findFundamentalMat(points1, points2, cv2.RANSAC)
    for i, inlier in enumerate(inliers):
        if inlier != 0:
            ransac_matches.append(matches[i])
    return ransac_matches
