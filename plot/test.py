import numpy as np
import struct

def read_binary_file(filename):
  ''' reads a binary file where the first 2x2bytes are num rows and columns
      the remainder of the file are float64 values '''
  f = open(filename, 'rb')
  rows = struct.unpack('i', f.read(4))[0]
  cols = struct.unpack('i', f.read(4))[0]
  return np.fromfile(f, dtype=np.float64).reshape((rows, cols))

def weighted_kde(particles, weights, bandwidth):
  def fn(x):
    return np.sum(weights * 1.0 / np.sqrt(2.0 * np.pi * bandwidth**2) * np.exp(-(particles-x)**2 / (2.0 * bandwidth))) / np.sum(weights)
  return fn

controls = read_binary_file('test.controls')

print controls

particles = read_binary_file('test.particles')
weights = read_binary_file('test.weights')

from matplotlib import pyplot as plt
for k in range(4):
  kde = weighted_kde(particles[:,k], weights[:,k], 0.1/(k+1))
  x = np.linspace(-3,3,201)
  y = map(kde, x)
  plt.plot(x, y)
  mean = np.sum(particles[:,k] * weights[:,k]) / np.sum(weights[:,k])
  # plt.hist(particles[:,k], 200, weights=weights[:,k], histtype='stepfilled', alpha=0.8)
plt.legend(['k=0','k=1','k=2','k=3'])
plt.show()

coefficients = read_binary_file('test.coefficients')
print coefficients
