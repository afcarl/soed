import numpy as np
import struct

def read_binary_file(filename):
  ''' reads a binary file where the first 2x2bytes are num rows and columns
      the remainder of the file are float64 values '''
  f = open(filename, 'rb')
  rows = struct.unpack('i', f.read(4))[0]
  cols = struct.unpack('i', f.read(4))[0]
  return np.fromfile(f, dtype=np.float64).reshape((rows, cols))

prefix = 'dp'

controls     = read_binary_file(prefix + '.controls')
particles    = read_binary_file(prefix + '.particles')
weights      = read_binary_file(prefix + '.weights')
coefficients = read_binary_file(prefix + '.coefficients')
means        = read_binary_file(prefix + '.trainingMeans')
variances    = read_binary_file(prefix + '.trainingVariances')
values       = read_binary_file(prefix + '.trainingValues')

print coefficients

stages       = len(controls) + 1

value_functions = [
  lambda mean, variance : 1,
  lambda mean, variance : mean,
  lambda mean, variance : mean * mean,
  lambda mean, variance : np.log(np.sqrt(variance)),
  lambda mean, variance : variance,
]

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

for k in range(1, stages - 1):
  fig = plt.figure()
  fig.set_size_inches(6, 6)
  ax = fig.add_subplot(111, projection='3d')
  xmin, xmax = -1, 1
  ymin, ymax = 10**(-3), 10**(0.25)
  nx, ny = 51, 51
  scaterplot = ax.scatter(means[:,k], np.log10(variances[:,k]), values[:,k], c='k', marker='o', s=2, alpha=0.5)
  X = np.linspace(xmin, xmax, nx)
  Y = np.logspace(np.log10(ymin), np.log10(ymax), ny, base=10)
  X, Y = np.meshgrid(X, Y)
  Z = np.zeros((nx, ny))
  for index, value_function in enumerate(value_functions):
    for i in range(nx):
      for j in range(ny):
        Z[i, j] += coefficients[index, k] * value_function(X[i, j], Y[i, j])
  surf = ax.plot_surface(X, np.log10(Y), Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.8, linewidth=0, antialiased=True)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(np.log10(ymin), np.log10(ymax))
  ax.set_zlim(np.min(Z) - (np.max(Z) - np.min(Z)) * 0.1, np.max(Z) + (np.max(Z) - np.min(Z)) * 0.1)
  ax.set_xlabel('mean')
  ax.set_ylabel('log10(variance)')
  cset = ax.contour(X, np.log10(Y), Z, 15, zdir='z', offset=ax.get_zlim()[0], cmap=cm.coolwarm)
  ax.yaxis.set_major_locator(LinearLocator(5))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax.zaxis.set_major_locator(LinearLocator(5))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  plt.savefig('%s_%d.pdf' % (prefix, k), bbox_inches='tight')
  # plt.show()

