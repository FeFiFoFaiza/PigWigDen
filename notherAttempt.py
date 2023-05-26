import cv2
import numpy as np
import matplotlib.pyplot as plt

dbSize = 30
shape = (112,92)
dim = shape[0]*shape[1]

P = np.empty(shape=(dim,30), dtype='float64')

# Read images into P
for i in range(dbSize):
    imgReading = plt.imread("./database/person" + str(i + 1) + ".pgm")
    P[:,i] = imgReading.reshape(dim)

# Find mean face maybe make my own mean function(?)
meanFace = P.mean(axis=1, keepdims=True)
plt.imshow(meanFace.reshape(shape), cmap="gray")
plt.show()

# Subtract mean face from each face
P -= meanFace

# Find covariance matrix by using transpose bc rank(cols) = rank(rows)
C = np.dot(P.T, P)/dbSize

# Find eigenvalues and eigenvectors
Vals, Vecs = np.linalg.eig(C)
inds = Vals.argsort()[::-1]
Vals = Vals[inds]
Vecs = Vecs[:,inds]

# Display eigenfaces
