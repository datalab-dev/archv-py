need python version 3
and opencv version 3.4

to install, can use unofficial opencv package from pypi
pip isntall opencv-python opencv-contrib-python
development of a python version of archv


filter.py

filter_keypoints(keypoints, min_size, min_response)

  filters out all the keypoints that aren't larger than 
  minimum size or response specified in the surf parameters

  input: keypoints list
  min_size (from surf parameters)
  min_response (from surf parameters)
  output: filtered list of keypoints


ratio_test(matches)

  taken from opencv3.4 documentation on feature matching in python
  https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

  only keep the matches where the best match is significantly
  different from the second best match. In this case, the best
  match is less than 80% of the second best match. The lower 
  ratio (r) the more matches will be filtered out

  input: matches list (the output of bf.knnMatch with k=2)
  output: matches list (all the matches that pass the ratio test)


symmetry_test(matches, matches2)



