import cv2
import matplotlib.pyplot as plt
import numpy as np


# Read the images
left = cv2.imread('sword1/im0.png', cv2.IMREAD_GRAYSCALE)
right = cv2.imread('sword1/im1.png', cv2.IMREAD_GRAYSCALE)

# Prepare the intrinsic parameters of the two cameras
# we assume no lens distortion, i.e. dist coeffs = 0
cameraMatrix1 = np.array([[6872.874, 0, 1605.291],
                          [0, 6872.874, 938.212],
                          [0, 0, 1]])
distCoeffs1 = np.array([0,0,0,0,0])

cameraMatrix2 = np.array([[6872.874, 0, 1922.709],
                          [0, 6872.874, 938.212],
                          [0, 0, 1]])
distCoeffs2 = np.array([0,0,0,0,0])

# Prepare the extrinsic parameters
rotationMatrix = np.eye(3) # we assume no rotation
transVector = np.array([-174.724, 0.0, 0.0])

# Rectify the images using both the intrinsic and extrinsic parameters
# Note: the image pair are already rectified so we do not really need
# to do this but we need to Q matrix to compute depth later on
image_size = left.shape[::-1]
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                  cameraMatrix2, distCoeffs2,
                                                  image_size, rotationMatrix, transVector)

# Remap the left image based on the resulting rotation R1 and projection matrix P1
leftmapX, leftmapY = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_32FC1)
left_remap = cv2.remap(left, leftmapX, leftmapY, cv2.INTER_LANCZOS4)

# Do the same for the right image
rightmapX, rightmapY = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_32FC1)
right_remap = cv2.remap(right, leftmapX, rightmapY, cv2.INTER_LANCZOS4)

# Let's try to compare the two images before and after rectifying them
# Plot the two original images side-by-side
ax = plt.subplot(211)
ax.set_title('original')
ax.imshow(np.hstack([left,right]),'gray')

# Plot the two remapped images side-by-side
ax = plt.subplot(212)
ax.set_title('remapped')
ax.imshow(np.hstack([left_remap,right_remap]),'gray')
plt.show()
# They look pretty much identical with the expection of some extra padding
# for the rectifying images. You can use both pair of images for the rest
# of the exercise. It is up to you - both should give valid results.

# You do the rest! Start by calculating the disparity map
