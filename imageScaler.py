import numpy

class imageScaler :
    
    height = 660
    width = 590
    
    def __singleImageScaling(self,inputimage):
        m = len(inputimage)
        n = len(inputimage[0])
        if self.height<m or self.width<n :
            raise ValueError('Target size smaller than image')
        output = numpy.zeros((self.height,self.width),dtype=numpy.uint8)
        output[(self.height-m)/2:(self.height+m)/2,(self.width-n)/2:(self.width+n)/2] = inputimage
        return output
        
        
    def scaleDataset(self,inputdata):
        n = len(inputdata)
        output = []
        try:
            for i in range(n):
                scaledImage = self.__singleImageScaling(inputdata[i][0])
                element = (scaledImage,inputdata[i][1])
                output.append(element)
        except ValueError as e:
            print(e)
        return output
