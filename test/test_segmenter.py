import cv2
from scipy import misc
from img_loading import ImageLoader
from preprocessing import segmenter

img = ImageLoader().loadImage('../data/part1/hand1_0_bot_seg_1_cropped.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binImg = segmenter.backgroundSubtract(grayImg)
misc.imsave('binImg.png', binImg)