# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:23:01 2016

@author: Jiyuan
"""

import numpy as np
import cv2
import pickle
from learning.knn_wrapper import KNNClassifier
from preprocessing import noise_remover, img_segmenting
from skimage import color
from feature.MomentsCalculator import MomentsCalculator

cv2.namedWindow("preview")
cv2.namedWindow("MOG")
cv2.namedWindow("ABLUR")
pkl_file = open('knn15.pkl','rb')
knn = pickle.load(pkl_file)
pkl_file.close()
vc = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG(200,5,0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    
str = "Hello World!"
bg = color.rgb2gray(frame)*255
segmenter = img_segmenting.ImgSegmenter(10)

while rval:
    cv2.rectangle(frame,(295,330),(1,1),(0,255,0),0)
    cv2.putText(frame, str, (400,400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255))
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    #fgmask = fgbg.apply(frame)
    fgmask = segmenter.backgroundSubtract2(color.rgb2gray(frame)*255, bg)
    cv2.imshow('MOG',fgmask)
    key = cv2.waitKey(20)
    if key == 32: # exit on ESC
        image = fgmask[1:331,1:296]
        imageresized = cv2.resize(image,(590,660),interpolation = cv2.INTER_CUBIC)
        image = noise_remover.Gaussian_filter_forImg(imageresized,3)
        cv2.imshow('ABLUR',image)
        input_x = np.asarray(MomentsCalculator().ImageMoments(image))
        output = knn.predict(input_x)
        str = output[0]
    if key == 27:
        break
vc.release()