import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
board_size = (9,6)
pts_obj = np.zeros((board_size[0]*board_size[1],3), np.float32)
pts_obj[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

# Scale the object points to correspond with the actual size of the squares
square_size = 20.0 # i.e. 20 mm
pts_obj *= square_size

# Arrays to store object points and image points from all the images.
all_pts_obj = [] # 3d point in real world space
all_pts_right_img = [] # 2d points in image plane.
all_pts_left_img = [] # 2d points in image plane.
images = glob.glob('opencv-samples/left/*.jpg')
image_names = []

# Loop through all the images
for fname in images:

    # Loop image and convert to grayscale
    left_img = cv2.imread(fname)
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    #right_img = cv2.imread(fname.replace('Left','Right'))
    right_img = cv2.imread(fname.replace('left','right'))
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    left_ret, left_corners = cv2.findChessboardCorners(left_gray, board_size, None)
    right_ret, right_corners = cv2.findChessboardCorners(right_gray, board_size, None)

    # If found in both images, add object points, image points (after refining them)
    if(left_ret and right_ret):
        image_names.append(fname)
        # Refine detected image points
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        left_corners2 = cv2.cornerSubPix(left_gray,left_corners, (11,11), (-1,-1), criteria)
        right_corners2 = cv2.cornerSubPix(right_gray,right_corners, (11,11), (-1,-1), criteria)

        # Store object points and image points for calibration later
        all_pts_obj.append(pts_obj)
        all_pts_left_img.append(left_corners2)
        all_pts_right_img.append(right_corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(left_img, board_size, left_corners2, left_ret)
        cv2.drawChessboardCorners(right_img, board_size, right_corners2, right_ret)
        #plt.imshow(np.hstack([left_img, right_img]))
        #plt.show()


# Calibrate both cameras seperately - often recommended
ret_left, K_left, dist_left, _, _ = cv2.calibrateCamera(all_pts_obj, all_pts_left_img,
                                                        left_gray.shape[::-1], None, None)
ret_right, K_right, dist_right, _, _ = cv2.calibrateCamera(all_pts_obj, all_pts_right_img,
                                                           right_gray.shape[::-1], None, None)


# Fix the intrinsic parameters as we have already calibrated them
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER +
                   cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(all_pts_obj,
                                                      all_pts_left_img,
                                                      all_pts_right_img,
                                                      K_left,
                                                      dist_left,
                                                      K_right,
                                                      dist_right,
                                                      left_gray.shape[::-1],
                                                      criteria_stereo, flags)

# You do the rest!
