import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


cap = cv.VideoCapture('Data/slow_traffic_small.mp4')
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
x, y, w, h = 591, 180, 40, 62 # simply hardcoded the values
track_window = (x, y, 20, 25)

# set up the ROI for tracking
roi = cv.imread('Data/biker.png')
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)    
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# cv.imshow('1', roi)
# cv.imshow('2', hsv_roi)
# cv.imshow('3', mask)
# cv.waitKey(0)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
counter = 0
while(1):
    ret, frame = cap.read()
    counter += 1
    
    if counter >= 117:
        if ret == True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            # apply meanshift to get the new location
            ret, track_window = cv.meanShift(dst, track_window, term_crit)
            # Draw it on image
            x,y,w,h = track_window
            img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            cv.imshow('img2',img2)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break
    else: 
        cv.imshow('img2',frame)
        cv.waitKey(30)