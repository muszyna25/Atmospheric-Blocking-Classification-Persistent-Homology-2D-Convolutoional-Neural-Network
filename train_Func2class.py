#!/usr/bin/env python
""" read input hd5 tensors
train net
write net + weights as HD5
"""

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
    parser.add_argument("--nCpu", type=int, default=0,
                        help="num CPUs used when fitting, use 0 for all resources")
    parser.add_argument("-n", "--events", type=int, default=0,
                        help="events for training, use 0 for all")

    parser.add_argument("-e", "--epochs", type=int, default=3,
                        help="fitting epoch")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="fit batch_size")
    parser.add_argument("--dropFrac", type=float, default=0.2,
                        help="drop fraction at all layers")
    parser.add_argument( "-s","--earlyStop", dest='earlyStopOn',
                         action='store_true',default=False,help="enable early stop")
    parser.add_argument( "--checkPt", dest='checkPtOn',
                         action='store_true',default=False,help="enable check points for weights")

    parser.add_argument( "--reduceLr", dest='reduceLearn',
                         action='store_true',default=False,help="reduce learning at plateau")
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
deep.load_input_hdf5(['train','val'])

#gra.plot_input_digit(deep.data['val'],range(4))
#gra.display_all('form')

deep.build_model(args) 
gra.plot_model(deep)
deep.train_model(args) 
deep.save_model_full() 
X,Yhot,Yprob=deep.make_prediction('val') 

gra.plot_train_hir(deep,args)

gra.plot_labeled_scores(Yhot,Yprob,'val',figId=-10)
gra.plot_AUC(Yhot,Yprob,'val',args,figId=-10)
gra.display_all('train')

