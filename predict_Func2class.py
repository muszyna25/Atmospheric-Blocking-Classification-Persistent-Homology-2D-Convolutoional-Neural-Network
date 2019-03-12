#!/usr/bin/env python
""" read input hd5 tensors
read trained net : model+weights
read test data from HD5
evaluate test data 
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from Plotter_Func2class import Plotter_Func2class
from Deep_Func2class import Deep_Func2class

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', dest='modelDesign',  default='cnn2d',
                        choices=['lstm','cnn1d','cnn2d'], help=" model design of the network")
    parser.add_argument("--funcDim", default='func2dim', 
                        choices=['func2dim','func1dim'], help="input funcis 1 or 2 dim histo")

    parser.add_argument("--dataPath",help="output path",  default='data')
    parser.add_argument("--outPath",
                        default='out',help="output path for plots and tables")
    parser.add_argument("-n", "--events", type=int, default=0,
                        help="events for training, use 0 for all")

    parser.add_argument('-X', "--no-Xterm", dest='noXterm',
                         action='store_true', default=False,
                         help="disable X-term for batch mode")
    args = parser.parse_args()
    args.prjName='func2class'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#=================================
#=================================
#  M A I N 
#=================================
#=================================
args=get_parser()

gra=Plotter_Func2class(args)
deep=Deep_Func2class(**vars(args))

dom='test'
#dom='val'

deep.load_input_hdf5([dom])
deep.load_model_full() 
X,Yhot,Yprob=deep.make_prediction(dom) 
gra.plot_labeled_scores(Yhot,Yprob,dom)
gra.plot_AUC(Yhot,Yprob,dom,args)

#deep.load_weights('weights_best') 
#deep.make_prediction(dom)

gra.display_all('predict')



