# for AUC of ROC
from sklearn.metrics import roc_curve, auc
import numpy as np
from keras.utils import plot_model


__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

#............................
#............................
#............................
class Plotter_Func2class(object):
    """Graphic ops related to formatin,training, eval of deep net """

    def __init__(self, args,xinch=8,yinch=7):
        if args.noXterm:
            print('diasable Xterm')
            import matplotlib as mpl
            mpl.use('Agg')  # to plot w/o X-server
        import matplotlib.pyplot as plt
        print(self.__class__.__name__,':','Graphics started')
        plt.close('all')
        self.outPath=args.outPath
        self.prjName=args.prjName

        self.plt=plt
        self.figL=[]
        self.outPath=args.outPath
        self.nr_nc=(3,3)

    #............................
    def display_all(self,pdf=1):
        if len(self.figL)<=0:
            print('display_all - nothing top plot, quit')
            return
        if pdf:
            for fid in self.figL:
                self.plt.figure(fid)
                self.plt.tight_layout()
                figName='%s/%s_%d'%(self.outPath,self.prjName,fid)
                print('Graphics saving to %s  ...'%figName)
                self.plt.savefig(figName+'.png')
        self.plt.show()

# figId=self.smart_append(figId)
#...!...!....................
    def smart_append(self,id): # increment id if re-used
        while id in self.figL: id+=1
        self.figL.append(id)
        return id

#............................
    def plot_model(self,deep):
        fname=self.prjName+'.graph.svg'
        plot_model(deep.model, to_file=fname, show_shapes=True, show_layer_names=True)
        print('Graph saved as ',fname)

#............................
    def plot_input_raw(self,deep,idxL,figId=5):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,4))
                 
        X=deep.raw['Xorg']
        Y=deep.raw['Yorg']
        nrow,ncol=2,3
        print('plot input raw for idx=',idxL)
        j=0
        for i in idxL:            
            #  grid is (yN,xN) - y=0 is at the top,  so dumm
            ax = self.plt.subplot(nrow, ncol, 1+j)
            tit='i:%d dig=%d'%(i,Y[i])           
            ax.set(title=tit)
            ax.imshow(X[i], cmap=self.plt.get_cmap('gray'))
            j+=1

#####........My code....................    
    def show_patch(self, M, figsizx, figsizy): #For example, y:[0:160] x:[0:320]
        plt.clf()
        plt.figure(figsize = (figsizx,figsizy))
        plt.imshow(M, interpolation='bilinear') #It does blinear interpolation of data to display image.
        plt.axis('off') #Turns off the ticks on both axises.
        plt.show()

#.......................................
    def plot_barcode_and_pers_dgm(self, dgms, indx):
        for dim in range(0, len(dgms)):
            dgm = dgms[dim]
            if dim==0:
                data = dgm[0:-1][:] #If H_0, it skips the bar with Inf.
            else:
                data = dgm[:] #Takes all data for the given H_N dimension.
    
            offset = 0 #Sets the vertical offset between bars (lines) for persistence barcode plot.
            no = data.shape[0] #Gets number of points (lines/bars).
    
            fig = plt.figure(figsize=(8,8)) #Sets a figure for both subplots.
    
            '''Plot persistence barcode.'''
            ax = fig.add_subplot(2,2,1)
            ax.autoscale(enable=True) #Adjusts the scale of axes automatically.
            ax.xaxis.set_major_formatter(FormatStrFormatter('%g')) #Adjusts the format of ticks in both automatically.
    
            for i in range(0, data.shape[0]):
                ax.hlines(y=0.1+offset, xmin=data[i][0], xmax=data[i][1], linestyle='-', linewidth=1, color='k') #Plots all horizontal lines/bars.
                offset += 0.09 #Shifts each line/bar by the fixed offset.
        
            plt.title('Barcode: H%i | No of bars: %i' %(dim, no))
            plt.yticks([]) #Sets no ticks on y-axis.
            plt.grid(True) #Sets grid on.
        
            fig.savefig('H0_Barcode_Persistence_Diagram_' + indx + '.png')
        
            '''Plot persistence diagram.'''
            ax = fig.add_subplot(2,2,2)
            ax.autoscale(enable=True)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            for i in range(0, data.shape[0]):
                ax.scatter(x=data[i][0], y=data[i][1], c='k') #Plots all points (x-birth of feature, y-death of feature) on the diagram.

            plt.title('Persistence diagram: H%i | No of points: %i' %(dim, no))
            plt.grid(True)

            plt.subplots_adjust(wspace=0.7) #Adjusts white space between the subplots.
            fig.show()
            fig.savefig('H1_Barcode_Persistence_Diagram_' + indx + '.png')

#............................
    def compute_histogram(self, dgms, indx):
        fig = plt.figure(figsize=(12,8))
        for dim in range(0, len(dgms)):
            dgm = dgms[dim]
            lbars = compute_bars_lengths(dgm)
            ax = fig.add_subplot(2,2,1+dim)
            n, bins, patches = plt.hist(x=lbars, bins=30, density=True, log=True, color='#0504aa', alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram for H%i' %dim)
            maxfreq = n.max()
            plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10) #Sets a clean upper y-axis limit.

#............................
    def plot_multiple_imgs(self, l_imgs, idxL, figId=9):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId, facecolor='white', figsize=(8,8))
        n=len(l_imgs); #no_imgs = int(np.round(n/2));
        j=0;  
        nrow,ncol=int(len(idxL)/2),2
        for i in idxL: 
            ax = self.plt.subplot(nrow, ncol, j+1)
            ax.imshow(l_imgs[i], interpolation='bilinear', cmap='viridis') #Plots multiple subimages next to each other.
            j+=1
    
#####........My code....................    

#............................
    def plot_input(self,XY,idxL,mode,figId=7):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,8))
        nrow,ncol=int(len(idxL)/2),2
        X=XY['X']; Y=XY['Y']
        print('plot input pos/neg for idx=',idxL)
        j=0
        col={0:'b',1:'g'}
        for i in idxL:
            x=X[i]; y=Y[i]
            #  grid is (yN,xN) - y=0 is at the top,  so dumm
            ax = self.plt.subplot(nrow, ncol, 1+j)
            tit='i:%d y=%d'%(i,y)           
            ax.set(title=tit)
            if mode=='digit':
                dig2D=x.reshape((28,28))
                ax.imshow(dig2D, cmap=self.plt.get_cmap('gray'))
            if mode=='func':
                ax.plot(x, alpha=0.5,c=col[y])
            j+=1


#............................
    def plot_train_hir(self,dee,args,figId=10): 
        figId=self.smart_append(figId)
        self.plt.figure(figId,facecolor='white', figsize=(11,6))
        nrow,ncol=self.nr_nc
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        ax1 = self.plt.subplot2grid((nrow,ncol), (1,0), colspan=2 )
        ax2 = self.plt.subplot2grid((nrow,ncol), (2,0), colspan=2,sharex=ax1 )
        ax3 = self.plt.subplot2grid((nrow,ncol), (0,0), colspan=2,sharex=ax1 )

        DL=dee.train_hirD
        val_acc=DL['val_acc'][-1]

        tit1='%s, train %.1f min, nCpu=%d, drop=%.1f'%(self.prjName,dee.train_sec/60.,args.nCpu,args.dropFrac)
        tit2='earlyStop=%d, end val_acc=%.3f'%(args.earlyStopOn,val_acc)

        
        ax1.set(ylabel='loss',title=tit1)
        ax1.plot(DL['loss'],'.-.',label='train')
        ax1.plot(DL['val_loss'],'.-',label='valid')
        ax1.legend(loc='best')
        ax1.grid(color='brown', linestyle='--',which='both')
        
        ax2.set(xlabel='epochs',ylabel='accuracy',title=tit2)
        ax2.plot(DL['acc'],'.-',label='train')
        ax2.plot(DL['val_acc'],'.-',label='valid')
        ax2.legend(loc='bottom right')
        ax2.grid(color='brown', linestyle='--',which='both')

        ax3.plot(DL['lr'],'.-',label='learn rate')
        ax3.legend(loc='best')
        ax3.grid(color='brown', linestyle='--',which='both')
        ax3.set_yscale('log')
        ax3.set(ylabel='learning rate')

#............................
    def plot_AUC(self,Yhot,Yprob,name,args,figId=20):
        if figId>0:
            self.figL.append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(5,9))
            nrow,ncol=2,1
            #  grid is (yN,xN) - y=0 is at the top,  so dumm
            ax1 = self.plt.subplot(nrow, ncol, 1)
            ax2 = self.plt.subplot(nrow, ncol, 2)
        
        else:
            self.plt.figure(-figId)
            nrow,ncol=self.nr_nc
            #  grid is (yN,xN) - y=0 is at the top,  so dumm
            ax1 = self.plt.subplot(nrow, ncol, nrow*ncol-ncol)
            ax2 = self.plt.subplot(nrow, ncol, nrow*ncol)


        # produce AUC of ROC
        '''
        With the Model class, you can use the predict method which will give you a vector of probabilities and then get the argmax of this vector (with np.argmax(y_pred1,axis=1)).
        '''
        # here I have only 1 class - so I can skip the armax step

        print('\nYprob',Yprob.shape,Yprob[:5])
        print('YHot (truth)',Yhot.shape,Yhot[:5])

        fpr, tpr, _ = roc_curve(Yhot,Yprob)
        roc_auc = auc(fpr, tpr)

        LRP=np.divide(tpr,fpr)
        fpr_cut=0.1
        for x,y in zip(fpr,LRP):
            if x <fpr_cut :continue
            print('found fpr=%.3f  LP+=%.3f  thr=%.3f'%(x,y,fpr_cut))
            break

        ax1.plot(fpr, tpr, label='ROC',color='seagreen' )
        ax1.plot([0, 1], [0, 1], 'k--', label='coin flip')
        ax1.axvline(x=x,linewidth=1, color='blue')
        ax1.set(xlabel='False Positive Rate',ylabel='True Positive Rate',title='ROC , area = %0.3f' % roc_auc)
        ax1.legend(loc='lower right',title=name+'-data')
        ax1.grid(color='brown', linestyle='--',which='both')

        ax2.plot(fpr,LRP, label='ROC', color='teal')
        ax2.plot([0, 1], [1, 1], 'k--',label='coin flip')
        ax2.set(ylabel='Pos. Likelih. Ratio',xlabel='False Positive Rate',title='LR+(FPR=%.2f)=%.1f'%(x,y))
        ax2.set_xlim([0,.1])

        ax2.axvline(x=x,linewidth=1, color='blue')
        ax2.legend(loc='upper right')
        ax2.grid(color='brown', linestyle='--',which='both')

        print('AUC: %f' % roc_auc)

#............................
    def plot_labeled_scores(self,Ytrue,Yscore,segName,score_thr=0.5,figId=21):
        if figId>0:
            self.figL.append(figId)
            fig=self.plt.figure(figId,facecolor='white', figsize=(7,4))
            ax=self.plt.subplot(1,1, 1)
        else:
            figId=-figId
            self.plt.figure(figId)
            nrow,ncol=self.nr_nc
            ax = self.plt.subplot(nrow, ncol, ncol)

        #print('nn',Yscore.ndim)
        assert Yscore.ndim==1
        u={0:[],1:[]}
        mAcc={0:0,1:0}
        for ygt,ysc in  zip(Ytrue,Yscore):
            u[ygt].append(ysc)
            if ysc> score_thr : mAcc[ygt]+=1
        
        mInp={0:len(u[0])+1e-3,1:len(u[1])+1e-3}

        print('Labeled scores found mAcc',mAcc, ' thr=',score_thr)

        bins = np.linspace(0.0, 1., 50)
        txt=''
        txt='TPR=%.2f, '%(mAcc[1]/mInp[1])
        ax.hist(u[1], bins, alpha=0.6,label=txt+'%d POS out of %d'%(mAcc[1],mInp[1]))
        txt='FPR=%.2f, '%(mAcc[0]/mInp[0])
        ax.hist(u[0], bins, alpha=0.5,label=txt+'%d NEG out of %d'%(mAcc[0],mInp[0]))

        ax.axvline(x=score_thr,linewidth=2, color='blue', linestyle='--')

        ax.set(xlabel='predicted score', ylabel='num samples')
        #ax.set_yscale('log')
        ax.grid(True)
        ax.set_title('Labeled scores dom=%s'%(segName))
        ax.legend(loc='upper right', title='score thr > %.2f'%score_thr)

  
