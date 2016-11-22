from scipy import ndimage

class NoiseRemoval():
    def Gaussian_filter(self, data, sig):
        res = []
        for item in data:
            img = item[0]
            img = ndimage.gaussian_filter(img, sigma=sig)
            label = item[1]
            res.append((img, label))
        return res

    def Median_filter(self, data):
        res = []
        for item in data:
            img = item[0]
            img = ndimage.median_filter(img)
            label = item[1]
            res.append((img, label))
        return res
