""" Method to test: python Test.py imageName """
import cv2
import preprocessing.NoiseRemoval
from scipy import ndimage
from preprocessing.img_segmenting import ImgSegmenter
from feature.Interests import Interest_points
import sys

image_name = str(sys.argv[1])
print image_name
image = ndimage.imread(image_name, mode='L')
cv2.imshow('image', image)
noise = NoiseRemoval.NoiseRemoval()
imgseg = ImgSegmenter(5)
interests = Interest_points()
image_seg = imgseg.backgroundSubtract(image)
cv2.imshow('image_seg', image_seg)
image_blur = noise.Gaussian_filter_forImg(image, 3)
cv2.imshow('image_blur', image_blur)
image_blur_seg = imgseg.binarizeImg(image_blur)
cv2.imshow('image_blur_seg', image_blur_seg) # blur then segmentation


kp_seg = interests.fast_feature(image_seg)
print "Total Keypoints without blur", len(kp_seg)
image_seg_kp = cv2.drawKeypoints(image_seg, kp_seg, color=(255, 0, 0))
cv2.imshow('image_seg_kp', image_seg_kp)
kp_blur_seg = interests.fast_feature(image_blur_seg)
print "Total Keypoints with blur", len(kp_blur_seg)
image_blur_seg_kp = cv2.drawKeypoints(image_blur_seg, kp_blur_seg, color=(255, 0, 0))
cv2.imshow('image_blur_seg_kp', image_blur_seg_kp)
raw_input('Press enter to close all images: ')
cv2.destroyAllWindows()
