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
args=get_parser()

'''
gra=Plotter_Func2class(args)

deep=Deep_Func2class(**vars(args))
deep.read_mnist_raw()
deep.select_digits_and_split([5,6]) # give 2 digits you want to train on
deep.save_input_hdf5()
gra.plot_input_raw(deep,range(6))
gra.plot_input(deep.data['val'],range(10),'digit',6)
if args.funcDim=='func1dim':
    gra.plot_input(deep.data['val'],range(10),'func',7)

gra.display_all('form')
'''

#................................

ph=Persist_Homologyclass(0)

ecmwf_data_path='/global/cscratch1/sd/muszyng/ethz_data/ecmwf_data/' # Name of one file that I have: ECMWF_1979_Jan.nc
varname = ['pv', 'z', 't', 'u']; y1=0; y2=100; x1=0; x2=100; indx = 0; timestep = 70; var_netcdf = np.array([]); norm = 'cosine'; maxdim = 1;
path = ecmwf_data_path
fns_list=ph.generate_list_of_files(path)
print(fns_list)

list_new_data = []
for i in range(0, len(fns_list)):
    fd=ph.read_netcdf_file(path, fns_list[i], varname[0])
    print('File:', fns_list[i])
    for j in range(0, len(fd)):
        print('Timestep: %d' %j)
        img = fd[j] #Gets an image from a file.
        img = ph.preprocessing_norm_stand(img) #Normalizes & standardizes data.
        l_imgs = ph.extract_subimages(img, 4) #Extracts eight subimages.
        for k in range(0, len(l_imgs)):
            #if k % 2 == 0:
            #I = add_rnd_noise(l_imgs[k]) #Adds randomness to create the second class of objects in the input raw data.    
            I = l_imgs[k]
            dgms = ph.PH_func_call(I, norm, maxdim) #Computes H1 homologies.
            new_repres_img = ph.hist_data(dgms) #Computes 2d histogram.
            if k%2 == 0:
                np.random.shuffle(new_repres_img) #Adds randomness to every second 2d histogram.
            list_new_data.append(new_repres_img)

