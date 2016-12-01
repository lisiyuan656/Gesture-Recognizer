# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:23:01 2016

@author: Jiyuan
"""

import cv2
from preprocessing import noise_remover, segmenter
from skimage import color
from preprocessing.process_img import process_img

cv2.namedWindow("preview")
cv2.namedWindow("MOG")
vc = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG(200,5,0.7)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.rectangle(frame,(295,330),(1,1),(0,255,0),0)
    cv2.putText(frame, "Hello World!", (400,400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255))
    cv2.imshow("preview", frame)
    fgmask = fgbg.apply(frame)
    cv2.imshow('MOG',fgmask)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 32: # exit on ESC
        image = frame[1:331,1:296]
        imageresized = cv2.resize(image,(590,660),interpolation = cv2.INTER_CUBIC)
        grayimage = color.rgb2gray(imageresized)*255
        image = noise_remover.Gaussian_filter_forImg(grayimage,3)
        binimage = segmenter.backgroundSubtract(image)
        input_x = process_img(binimage)
        #put input_x into trained nl or something else
        cv2.imshow("preview",imageresized)
    if key == 27:
        break
vc.release()