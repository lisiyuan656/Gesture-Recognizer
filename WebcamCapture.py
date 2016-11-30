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
    cv2.rectangle(frame,(295,330),(1,1),(0,255,0),0)
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 32: # exit on ESC
        image = frame[1:331,1:296]
        imageresized = cv2.resize(image,(590,660),interpolation = cv2.INTER_CUBIC)
        grayimage = color.rgb2gray(imageresized)*255
        image = noise_remover.Gaussian_filter_forImg(grayimage,3)
        binimage = segmenter.backgroundSubtract(image)
        
        cv2.imshow("preview",imageresized)
    if key == 27:
        break
vc.release()