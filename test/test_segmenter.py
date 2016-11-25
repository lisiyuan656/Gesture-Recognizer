from scipy import misc
from img_loading import ImageLoader
from preprocessing.img_segmenting import ImgSegmenter

img = ImageLoader().loadImage('../data/part1/hand1_0_bot_seg_1_cropped.png')
segmenter = ImgSegmenter()
binImg = segmenter.binarizeImg(img)
misc.imsave('binImg.png', binImg)