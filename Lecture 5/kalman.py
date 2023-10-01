import cv2 as cv
import argparse
import sys
import math
import numpy as np

def center(points):
    x = np.float32(
        (points[0][0] +
         points[1][0] +
         points[2][0] +
         points[3][0]) /
        4.0)
    y = np.float32(
        (points[0][1] +
         points[1][1] +
         points[2][1] +
         points[3][1]) /
        4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)

cap = cv.VideoCapture('Data/slow_traffic_small.mp4')
template = cv.imread('Data/biker.png')

hsv_template = cv.cvtColor(template, cv.COLOR_BGR2HSV)    

mask = cv.inRange(hsv_template, np.array((0., 50., 30.)), np.array((180., 255., 255.)))

template_hist = cv.calcHist([hsv_template], [0, 1], mask, [180, 256] , [0, 180, 0, 256])
cv.normalize(template_hist, template_hist, 0, 255, cv.NORM_MINMAX)


#Initialize KalmanFilter Object
kalman = cv.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

kalman_prediction = np.zeros((2, 1), np.float32)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

track_window = (592, 180, template.shape[1], template.shape[0])
counter=0

while True:
    _, frame = cap.read()
    counter += 1
    
    if counter < 114:
        continue

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    dst = cv.calcBackProject([hsv], [0, 1], template_hist, [0, 180, 0, 256], 1)
    ret, track_window = cv.CamShift(dst, track_window, term_crit)

    cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)

    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    kalman.correct(center(pts))
    kalman_prediction = kalman.predict()

    x, y, w, h = track_window
    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    frame = cv.circle(frame, (int(kalman_prediction[0]), int(kalman_prediction[1])), 3, (0, 255, 0), -1)
            
    cv.imshow('img2', frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
