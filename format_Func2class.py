#!/usr/bin/env python
"""  read raw input MNIST data , picks 2 digits only, converts 28x28 , optional in to flat 1d arrays of numbers - like a histograms.

write 6 tensors: (train,val,test) * (X,Y) in hd5

"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from Plotter_Func2class import Plotter_Func2class
from Deep_Func2class import Deep_Func2class
from Persist_Homologyclass import Persist_Homologyclass

import numpy as np
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath",help="output path",  default='data')
    parser.add_argument("--outPath",
                        default='out',help="output path for plots and tables")

    parser.add_argument("-n", "--events", type=int, default=0,
                        help="events for training, use 0 for all")

    parser.add_argument("--funcDim", default='func2dim', 
                        choices=['func2dim','func1dim'], help="input funcis 1 or 2 dim histo")

    parser.add_argument( '-X',"--no-Xterm", dest='noXterm',
                         action='store_true', default=False,
                         help="disable X-term for batch mode")
    args = parser.parse_args()
    args.prjName='func2class'
    args.modelDesign='xxx'	
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#=================================
#=================================
#  M A I N 
#=================================
#=================================

#............................
args=get_parser()

#............................
gra=Plotter_Func2class(args)

#............................
ecmwf_data_path='/global/cscratch1/sd/muszyng/ethz_data/ecmwf_data/' # Name of one file that I have: ECMWF_1979_Jan.nc
varname='pv'
ecmwf_labeled_data_path='/global/cscratch1/sd/muszyng/ethz_data/labels/'

#............................
ph=Persist_Homologyclass(ecmwf_data_path, ecmwf_labeled_data_path, varname) # You can specify max homology dimension and metric for distance matrix.

ph.generate_list_of_files()
ph.generate_data_list()
ph.generate_labeled_data_list()
print(ph.Y_labels)

#............................
#outputListData=[np.random.random((100, 100)) for i in range(1888)] # List of fake 2d histograms (10x10). 
#ph.save_dict_to_hdf5(outputListData)
#ph.load_hdf5_file('train.hd5')
#gra.plot_global_img(ph.l_globalimgs, 0)
#gra.plot_multiple_imgs(ph.l_imgs, range(8))
#gra.plot_multiple_histograms(0, ph.l_dgms, range(8)) 
#gra.plot_multiple_histograms(1, ph.l_dgms, range(8)) 
#gra.plot_multiple_barcodes(1, ph.l_dgms, range(8))
#gra.plot_multiple_diagrams(1, ph.l_dgms, range(8))

#gra.plot_global_img(ph.labels_l_globalimgs, 0)
#gra.plot_multiple_imgs(ph.labels_l_imgs, range(8))

gra.display_all('form') #This is to plot all collected figures.

#............................













'''
deep=Deep_Func2class(**vars(args))
deep.read_mnist_raw()
deep.select_digits_and_split([5,6]) # give 2 digits you want to train on
deep.save_input_hdf5()
gra.plot_input_raw(deep,range(6))
gra.plot_input(deep.data['val'],range(10),'digit',6)
if args.funcDim=='func1dim':
    gra.plot_input(deep.data['val'],range(10),'func',7)
gra.display_all('form') #This is to plot all collected figures.
'''

