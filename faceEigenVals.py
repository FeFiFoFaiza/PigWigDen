# MUST FINISH + FIX

import matplotlib.pyplot as plt
import numpy as np


dbSize = 30
P = np.array([])

for i in range(dbSize + 1):
    imgReading = plt.imread("./database/person" + str(i + 1) + ".pgm")
    m,n = imgReading.shape
    print(imgReading.reshape(m*n,1)[:,0])
    if i == 0:
        P = imgReading.reshape(m*n,1)[:,0]
    else:
        P = np.vstack((P, imgReading.reshape(m*n,1)[:,0]))

P = P.T #To transpose or not to tranpose, that is the question
print(P)

# Find mean face
# Implement my own mean function(?)
meanFace = np.mean(P, axis=1)
plt.imshow(meanFace.reshape(m,n), cmap="gray")
plt.show()

# P -= meanFace

#Subtract mean face from each face
A = np.array([])
for i in range(dbSize):
    if i == 0:
        A = P[:,i] - meanFace
    else:
        A = np.vstack((A, P[:,i] - meanFace))

# # Find covariance matrix
C = np.dot(A, A.T)
print(C.shape)

#Find eigenvalues and eigenvectors
Vals, Vecs = np.linalg.eig(C)
EigVecs = np.dot(A.T, Vecs)

# Normalize eigenvectors
# for i in range(dbSize):
#     EigVecs[:,i] = EigVecs[:,i]/np.linalg.norm(EigVecs[:,i])

for i in range(dbSize - 2):
    if i == 0:
        EigenFaces = np.reshape(EigVecs[:,i] + meanFace, (m,n))
    else:
        EigenFaces = np.vstack((EigenFaces, np.reshape(EigVecs[:,i] + meanFace, (m,n))))

# Show eigenfaces


# # show eigenfaces
EigenFaces = np.uint8(EigenFaces)
for i in range(dbSize):
    plt.subplot(6,5,i+1)
    print(i)
    plt.imshow(EigVecs[:,i].reshape(m,n), cmap="gray")
plt.show()
# EigenFaces = np.uint8(EigenFaces)
# plt.figure()
# plt.imshow(EigenFaces)
# plt.show()

# # I am an idiot why am i trying to do what I do
# # 10304 x 10304 ????
    
