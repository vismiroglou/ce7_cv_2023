import numpy as np
import cv2 as cv


cap = cv.VideoCapture('Data/slow_traffic_small.mp4')

# set up the ROI for tracking
roi = cv.imread('Data/biker.png')
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)    
mask = cv.inRange(hsv_roi, np.array((0., 50.,30.)), np.array((180.,255.,255.)))

roi_hist = cv.calcHist([hsv_roi], [0, 1], mask, [180, 256] , [0, 180, 0, 256])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)


# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
counter = 0
track_window = (592, 180, roi.shape[1], roi.shape[0])
while(1):
    ret, frame = cap.read()
    counter += 1
    if counter < 114:
        continue

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)
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