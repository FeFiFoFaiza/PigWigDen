import numpy as np
import os, cv2
import matplotlib.pyplot as plt


# Double Threshold
def canny(img, thresLow, thresHigh):

    # Turn pic grey
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Filter to reduce noise
    img = cv2.GaussianBlur(img, (5,5), 1.4)

    # Use Sobel Filter to get gradient in x and y direction
    gradX = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gradY = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # Turn to polar???? 
    magnitude, angle = cv2.cartToPolar(gradX, gradY, angleInDegrees=True)

    maxMag = np.max(magnitude)
    if not thresLow:
        thresLow = maxMag * 0.1
    if not thresHigh:
        thresHigh = maxMag * 0.5

    # Dimensions
    height, width = img.shape

    # Going thru every pixel
    for i in range(width):
        for j in range(height):

            gradAng = angle[j, i]
            if abs(gradAng) > 180:
                gradAng = abs(gradAng - 180)
            else:
                gradAng = abs(gradAng)
            
            # Go in the angle of gradient to next pixel

            # Angle between 0 and 22.5 -> 0
            if (gradAng < 22.5):
                prevPixX, prevPixY = i-1, j
                nextPixX, nextPixY = i+1, j

            # Angle between 22.5 and 67.5 -> 45
            elif (gradAng >= 22.5 and gradAng < 67.5):
                prevPixX, prevPixY = i-1, j-1
                nextPixX, nextPixY = i+1, j+1
            
            # Angle between 67.5 and 112.5 -> 90
            elif (gradAng >= 67.5 and gradAng < 112.5):
                prevPixX, prevPixY = i, j-1
                nextPixX, nextPixY = i, j+1

            # Angle between 112.5 and 157.5 -> 135
            elif (gradAng >= 112.5 and gradAng < 157.5):
                prevPixX, prevPixY = i-1, j+1
                nextPixX, nextPixY = i+1, j-1
                
            # non max supression
            if width>prevPixX>=0 and height>prevPixY>=0 and width>nextPixX>=0 and height>nextPixY>=0:
                if magnitude[j, i] < magnitude[prevPixY, prevPixX] or magnitude[j, i] < magnitude[nextPixY, nextPixX]:
                    magnitude[j, i] = 0
                    continue

            if width>nextPixX>=0 and height>nextPixY>=0:
                if magnitude[j, i] < magnitude[nextPixY, nextPixX]:
                    magnitude[j, i] = 0
                    

    # Double Threshold
    # Categories: strong, weak
    strong = np.zeros_like(img)
    weak = np.zeros_like(img)
    categories = np.zeros_like(img)

    for i in range(width):
        for j in range(height):

            gradintMag = magnitude[j, i]

            if gradintMag < thresLow:
                magnitude[j, i] = 0
            elif thresHigh>gradintMag>=thresLow:
                categories[j, i] = 1
            else:
                categories[j, i] = 2 #weak or strong?

    return magnitude


img = cv2.imread('Bird_Demo.jpg')

# Canny Edge Detection
cannyEdge = canny(img, None , None)

plt.figure()
f, plots = plt.subplots(2, 1)
plots[0].imshow(img)
plots[1].imshow(cannyEdge)

            


            




