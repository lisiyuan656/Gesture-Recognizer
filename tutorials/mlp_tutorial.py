import neurolab as nl
import numpy as np

x = np.linspace(-7, 7, 20)
y = np.sin(x) * 0.5
size = len(x)
inp = x.reshape(size,1)
tar = y.reshape(size,1)

# Input is in range [-7,7]. Input layer has 1 neuron.
# Hidden layer has 5 neurons. Output layer has 1 neuron.
net = nl.net.newff([[-7, 7]],[5, 1])
error = net.train(inp, tar, epochs=500, show=100, goal=0.02)
out = net.sim(inp)

# Plot result
import pylab as pl
pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')

x2 = np.linspace(-6.0,6.0,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = out.reshape(size)
pl.subplot(212)
pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pl.legend(['train target', 'net output'])
pl.show()