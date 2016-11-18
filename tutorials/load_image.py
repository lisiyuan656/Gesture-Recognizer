import os
from scipy import misc
import matplotlib.pyplot as plt

print os.path.isfile("../data/part1/hand1_0_bot_seg_1_cropped.png")
test = misc.imread('../data/part1/hand1_0_bot_seg_1_cropped.png')
plt.imshow(test)
plt.show()