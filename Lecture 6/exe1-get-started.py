import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the read and left stereo image pair
imgR = cv2.imread('tsukuba/scene1.row3.col3.ppm', cv2.IMREAD_GRAYSCALE)
imgL = cv2.imread('tsukuba/scene1.row3.col1.ppm', cv2.IMREAD_GRAYSCALE)


cv2.imshow('1', imgR)
cv2.waitKey(0)

# Calculate disparity map
stereo = cv2.StereoBM.create(numDisparities=32, blockSize=11)
disparity = stereo.compute(imgL,imgR)
plt.title('disparity map')
plt.imshow(disparity,'gray')
plt.show()
