from hist_equalization import *

def main():
    # alternatively: fixed = cv2.equalizeHist(img)
    img = cv2.imread("../before.jpg", 0)
    hist = make_histogram(img)
    transform = get_transform(hist)
    fixed = convert_image(img, hist, transform)

    res = np.hstack((img,fixed)) #stacking images side-by-side

    cv2.imshow('before-fixed', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
