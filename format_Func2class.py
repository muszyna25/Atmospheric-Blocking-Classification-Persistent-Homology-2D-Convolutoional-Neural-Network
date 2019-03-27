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
varname='t'
ecmwf_labeled_data_path='/global/cscratch1/sd/muszyng/ethz_data/labels/'

#............................
ph=Persist_Homologyclass(ecmwf_data_path, ecmwf_labeled_data_path, varname) # You can specify max homology dimension and metric for distance matrix.

'''These two are for testing'''
#out=ph.read_netcdf_file_('/global/cscratch1/sd/muszyng/ethz_data/ecmwf_download/batch_scripts/', 'ERA_INTERIM_1979.nc', 'pv') #Variables names: e.g., 'lon', 'lat', 'prw'
#outputListData=[np.random.random((100, 100)) for i in range(1888)] # List of fake 2d histograms (10x10). 

#............................
#print(out.shape)

ph.generate_list_of_files()
ph.generate_data_list()
ph.generate_labeled_data_list()

#............................
'''These two need to be fixed'''
#ph.save_dict_to_hdf5(ph.outputData)
#ph.load_hdf5_file('train.hd5')

''' Fixed plotting global binary mask and submasks'''
gra.plot_global_binary_mask(ph, 0)
gra.plot_multiple_binary_masks(ph, range(8))
#'''

''' Fixed plotting global image and subimages'''
gra.plot_global_img(ph, 0)
gra.plot_multiple_imgs(ph, range(8))
#'''

''' Fixed plotting histograms
gra.plot_multiple_1d_histograms(ph, range(8)) 
gra.plot_multiple_2d_histograms(ph, range(8)) 
'''

''' Fixed plotting barcodes and diagrams
gra.plot_multiple_barcodes(0, ph, range(8))
gra.plot_multiple_barcodes(1, ph, range(8))
gra.plot_multiple_diagrams(1, ph, range(8))
'''

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

