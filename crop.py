import cannyEdgeDetection as ced
import numpy as np
import cv2
import matplotlib.pyplot as plt

def crop(img):
    CannyPhoto = ced.canny(img, None, None)
    pts = np.argwhere(CannyPhoto>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite("cropped.png", cropped)

cv2.CascadeClassifier.detectMultiScale(cv2.cv.fromarray(ced.canny(cv2.imread("Bird_Demo.jpg"), None, None)))