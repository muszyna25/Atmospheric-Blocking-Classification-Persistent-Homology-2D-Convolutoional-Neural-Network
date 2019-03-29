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
data_path='/global/cscratch1/sd/muszyng/ethz_data/ecmwf_data/' # Path to netCDF files directory.
bin_mask_path='/global/cscratch1/sd/muszyng/ethz_data/labels/' # Path to binary mask files directory.

#............................
ph=Persist_Homologyclass(data_path, bin_mask_path) # Optional: 1) var name; 2) metric pairwise dist matrix; 3) max homology group dim.

ph.generate_dataset() # Generate dataset: (X - features, Y - labels).
#ph.save_dataset_hdf5() # Save dataset to hdf5 format: (train, val, test).

#............................
#ph.load_hdf5_file('train.hd5')

''' Fixed plotting global binary mask and submasks'''
#gra.plot_global_binary_mask(ph, 3)
#gra.plot_multiple_binary_masks(ph, range(24,32))
#'''

''' Fixed plotting global image and subimages'''
#gra.plot_global_img(ph, 0)
#gra.plot_multiple_imgs(ph, range(8))
#'''

''' Fixed plotting histograms
gra.plot_multiple_1d_histograms(ph, range(8)) 
gra.plot_multiple_2d_histograms(ph, range(8)) 
'''

''' Fixed plotting barcodes and diagrams'''
#gra.plot_multiple_barcodes(0, ph, range(8))
#gra.plot_multiple_barcodes(1, ph, range(8))
#gra.plot_multiple_diagrams(1, ph, range(8))

#gra.display_all('form') #This is to plot all collected figures.

#............................











'''These two are for testing'''
#out=ph.read_netcdf_file_('/global/cscratch1/sd/muszyng/ethz_data/ecmwf_download/batch_scripts/', 'ERA_INTERIM_1979.nc', 'pv') #Variables names: e.g., 'lon', 'lat', 'prw'
#outputListData=[np.random.random((100, 100)) for i in range(1888)] # List of fake 2d histograms (10x10). 

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

