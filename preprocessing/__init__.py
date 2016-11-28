from preprocessing.img_segmenting import ImgSegmenter
from preprocessing.NoiseRemoval import NoiseRemoval
from preprocessing.imageScaler import imageScaler

noise_remover = NoiseRemoval()
img_scaler = imageScaler()
segmenter = ImgSegmenter(5)
