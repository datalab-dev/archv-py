import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_histogram(image):
    """ converts image to histogram of pixel intensities """
    w,h = image.shape
    histogram = [0 for i in range(256)] 

    for i in range(w):
        for j in range(h):
            value = image[i,j]
            histogram[value] = histogram[value] + 1
    return [v /(w*h) for v in histogram]

def get_transform(hist):
    """ takes a histogram and equalizes it """
    cs = []
    total = 0
    for v in hist:
        total = total + v
        cs.append(total)

    return [v * 255 for v in cs]

def convert_image(img, hist, transform):
    fixed = np.zeros_like(img)
    w,h = img.shape

    for i in range(w):
        for j in range(h):
            fixed[i,j] = transform[img[i,j]]
    return fixed

def plot_histogram(hist):
    plt.bar(range(256), hist, color="blue")
    plt.title('image hist');
    plt.show()
