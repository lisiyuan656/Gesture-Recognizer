import numpy
def process_label(label):
  #output_y = numpy.zeros((36,1))
  if ord(label)<58:
      #output_y[ord(label)-48] = 1.
      output_y = ord(label)-48
  else:
      #output_y[ord(label)-87] = 1.
      output_y = ord(label)-87
  return output_y
