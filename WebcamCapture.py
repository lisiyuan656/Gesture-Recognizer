# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:23:01 2016

@author: Jiyuan
"""

import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.rectangle(frame,(590,660),(1,1),(0,255,0),0)
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
vc.release()
cv2.destroyWindow("preview")