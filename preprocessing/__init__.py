from preprocessing.img_segmenting import ImgSegmenter
from preprocessing import NoiseRemoval
from preprocessing import imageScaler

noise_remover = NoiseRemoval()
img_scaler = imageScaler()
segmenter = ImgSegmenter(5)