import numpy as np
import struct

def read_binary_file(filename):
  ''' reads a binary file where the first 2x2bytes are num rows and columns
      the remainder of the file are float64 values '''
  f = open(filename, 'rb')
  rows = struct.unpack('i', f.read(4))[0]
  cols = struct.unpack('i', f.read(4))[0]
  return np.fromfile(f, dtype=np.float64).reshape((rows, cols))

def weighted_kde(particles, weights, width):
  def fn(x):
    return 0.1 * np.sum(weights / np.sqrt(2.0 * np.pi * width**2) * np.exp(-(particles-x)**2 / (2.0 * width))) / np.sum(weights)
  return fn

prefix = 'dp'

controls     = read_binary_file(prefix + '.controls')
particles    = read_binary_file(prefix + '.particles')
weights      = read_binary_file(prefix + '.weights')
coefficients = read_binary_file(prefix + '.coefficients')

print controls
print coefficients

import matplotlib
matplotlib.use('PDF')

from matplotlib import pyplot as plt

matplotlib.rc('font', size='10')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('font' ,serif='Computer Modern')
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}'

colors = ['#ff0000','#ff9900','#99cc33','#33cc66','#0066ff']
bandwidths = [0.01, 0.01, 0.005, 0.003, 0.001]


fig = plt.figure()
fig.set_size_inches(5, 4)

for k in range(len(controls) + 1):
  kdefn = weighted_kde(particles[:,k], weights[:,k], bandwidths[k])
  x = np.linspace(-1,2,301)
  plt.plot(x, map(kdefn, x), color=colors[k])
  # mean = np.sum(particles[:,k] * weights[:,k]) / np.sum(weights[:,k])
plt.legend(['$k = 0$','$k = 1$','$k = 2$','$k = 3$','$k = 4$'])
plt.ylabel(r'Posterior density')
plt.xlabel(r'$\theta$ or $d$')

for k in range(len(controls)):
  kdefn = weighted_kde(particles[:,k], weights[:,k], bandwidths[k])
  plt.plot(controls[k], kdefn(controls[k]), 'o', color=colors[k], mew=0)

plt.savefig(prefix + '.pdf', bbox_inches='tight')

