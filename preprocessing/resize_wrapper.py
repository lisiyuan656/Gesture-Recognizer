# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:00:48 2016

@author: Jiyuan
"""

import cv2

def resize_wrapper(dataset,x):
    output = []
    for t in dataset:
        temp = cv2.resize(t[0],None,fx = 1./x,fy = 1./x, interpolation = cv2.INTER_CUBIC)
        output.append((temp,t[1]))
    return output