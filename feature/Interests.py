import numpy as np
import cv2

class Interest_points():
    def fast_feature(self, img):
        fast = cv2.FastFeatureDetector()
        kp = fast.detect(img, None)
        #print "Total Keypoints with nonmaxSuppression", len(kp)
        return len(kp)

    def get_min_max_distance(self, img, num_minmax):
        fast = cv2.FastFeatureDetector()
        fast.setBool('nonmaxSuppression',0)
        kp = fast.detect(img, None)
        M = cv2.moments(img)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        distance = []
        for i in range(len(kp)):
            distance.append(np.linalg.norm(np.asarray(kp[i].pt) - np.asarray((cx,cy))))
        distance.sort()
        res = np.array([])
        if (len(kp)==0):
            return np.zeros(num_minmax*2)
        for i in range(num_minmax):
            res = np.append(res, distance[i%len(kp)])
        for i in range(num_minmax):
            res = np.append(res, distance[(len(kp)-1-i)%len(kp)])
        return res
