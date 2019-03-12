from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

data = datasets.make_circles(n_samples=100)[0] + 5 * datasets.make_circles(n_samples=100)[0]

flag_cocycs = False

result = ripser(data, do_cocycles=flag_cocycs)
dgms = result['dgms']
print(dgms[0][1:10])

if flag_cocycs:
	cocycs = result['cocycles']
	print(cocycs[1])

plot_diagrams(dgms, show=True)

### TO DO: 
'''
1) Create function to plot barcode
2) Create function to calculate length of bars
3) Create function to plot histogram
4) Create function to save input data in hdf5 format
'''

