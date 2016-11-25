import numpy as np
import cv2

class Interest_points():
    def fast_feature(self, img):
        fast = cv2.FastFeatureDetector()
        kp = fast.detect(img, None)
        print "Total Keypoints with nonmaxSuppression", len(kp)
        return kp
