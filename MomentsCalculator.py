import cv2
import math

class MomentsCalculator:
    
    def __ImageMoments(self,inputimage):
        M = cv2.moments(inputimage)
        eta02 = M['mu02']/math.pow(M['m00'],2)
        eta03 = M['mu03']/math.pow(M['m00'],2.5)
        eta11 = M['mu11']/math.pow(M['m00'],2)
        eta12 = M['mu12']/math.pow(M['m00'],2.5)
        eta20 = M['mu20']/math.pow(M['m00'],2)
        eta21 = M['mu21']/math.pow(M['m00'],2.5)
        eta30 = M['mu30']/math.pow(M['m00'],2.5)
        return (eta02,eta03,eta11,eta12,eta20,eta21,eta30)
        
    
    def Moments(self,inputdata):
        n = len(inputdata)
        output = []
        for i in range(n):
            tempmoments = self.__ImageMoments(inputdata[i][0])
            output.append(tempmoments)
        return output