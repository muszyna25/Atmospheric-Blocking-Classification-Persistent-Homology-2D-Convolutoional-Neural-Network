import os, time
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

start = time.time()

from keras.datasets import mnist
from keras import utils as np_utils
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau
from keras.layers import Dense, Dropout,   Input, Conv1D,MaxPool1D,Flatten,Reshape,LSTM, Conv2D,MaxPool2D

import numpy as np
import h5py
print('deep-libs imported elaT=%.1f sec'%(time.time() - start))

#............................
from keras.callbacks import Callback
import keras.backend as K
class MyLearningTracker(Callback):
    def __init__(self):
        self.hir=[]   
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        #lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        lr = K.eval(optimizer.lr)
        self.hir.append(lr)


#............................
#............................
#............................
class Deep_Func2class(object):

    def __init__(self,**kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
 
        for xx in [ self.dataPath, self.outPath]:
            if os.path.exists(xx): continue
            print('Aborting on start, missing  dir:',xx)
            exit(99)

        print(self.__class__.__name__,'TF ver:', K.tf.__version__,', prj:',self.prjName)
        self.data={}
        ''' data container holds only X-values
        data[dom][X,Y], where dom=train/val/test, X=func, Y=0/1
        '''

#............................
    def read_mnist_raw(self):
        print('read raw data')
        # Load pre-shuffled MNIST data into train and test sets
        (X, Y), (Xt, Yt) = mnist.load_data()
        self.raw={}
        self.raw['Xorg']=np.concatenate((X,Xt))
        self.raw['Yorg']=np.concatenate((Y,Yt))
        print('raw MNIST Xorg sum',self.raw['Xorg'].shape)
    
    #............................
    def select_digits_and_split(self,digL):
        assert len(digL)==2
        print('select_digits:',digL)
        #self.roleL={'neg':digL[0],'pos':digL[1]} #???
        
        trainFrac=0.7; valFrac=0.1
        prob2=trainFrac+valFrac
        assert prob2 <1
        assert trainFrac <prob2

        self.data={dom:{'X':[],'Y':[]} for dom in ['train','val','test']}
 
        n=0
        for x,d in zip(self.raw['Xorg'],self.raw['Yorg']):
            if n > self.events and self.events>0 : break
            if d not in digL: continue # skip  not used digits
            n+=1
            if self.funcDim=='h1dim':
                xf=x.flatten().astype(float)
            else:
                xf=x.astype(float)
            sum=np.sum(xf)/100.
            xf/=sum

            y= d==digL[1] # binary label 

            r=np.random.uniform()
            if r <trainFrac :
                dom='train'
            elif r <prob2 :
                dom='val'
            else:
                dom='test'
            self.data[dom]['X'].append(xf)
            self.data[dom]['Y'].append(y)
            
        print('split summary for ',self.data.keys())
        for dom in self.data:
            ddd=self.data[dom]
            for xy in ddd:
                ddd[xy]=np.array(ddd[xy]).astype(float)
                if xy=='X':
                    print('split for dom=',dom,xy,ddd[xy].shape)
                if xy=='Y':
                    n01=ddd[xy].shape[0]
                    n1=np.sum(ddd[xy])
                    bal=n1/n01
                    print('    ',dom,'Y balance:%.3f'%bal)

#............................
    def save_input_hdf5(self):
        for dom in self.data:
            outF=self.dataPath+'/'+self.prjName+'_%s.%s.hd5'%(dom,self.funcDim)
            #print('save data as hdf5:',outF)
            h5f = h5py.File(outF, 'w')
            for xy in self.data[dom]:
                xobj=self.data[dom][xy]
                h5f.create_dataset(xy, data=self.data[dom][xy])
            h5f.close()
            xx=os.path.getsize(outF)/1048576
            print('closed  hdf5:',outF,' size=%.2f MB'%xx)

#............................
    def load_input_hdf5(self,domL):
        for dom in domL:
            #inpF=self.dataPath+'/'+self.prjName+'_%s.%s.hd5'%(dom,self.funcDim)
            #if self.dataPath=='data2':
            #if self.dataPath=='':
            inpF=self.dataPath+'/%s.hd5'%(dom)
            #inpF=self.dataPath+'/tzfunc2class_%s.hd5'%(dom)

            print('load hdf5:',inpF)
            h5f = h5py.File(inpF, 'r')
            self.data[dom]={}
            for xy in h5f.keys(): 
                npA=h5f[xy][:]
                # this code only reduces the size of input - if requested
                if self.events>0:
                    mxe=self.events 
                    if dom!='train':  mxe=int(mxe/5) 
                    pres=int(npA.shape[0]/ mxe)
                    if pres>1: npA=npA[::pres]
                    print('reduced ',dom,xy,' to %d events'%npA.shape[0])
                self.data[dom][xy] = npA 
                print(' done',dom,xy,self.data[dom][xy].shape)
            h5f.close()
        print('load_input_hdf5 done, elaT=%.1f sec'%(time.time() - start))

    #............................
    def print_input(self,name,k=2):
        for x in self.data:
            if name not in x:  continue
            xobj=self.data[x]
            print('\nsample of ',x, xobj.shape)
            for i in range(k):
                if 'Y' in x:
                    print('\nidx=%d digit=%d, X-data:'%(i,xobj[i]))
                else:
                    print('\n%d data:'%i)
                    print(xobj[i][5:7])

    #............................
    def build_model(self,args):
        # based  https://keras.io/getting-started/functional-api-guide/
        start = time.time()
        # CPUs are used via a "device" which is just a threadpool
        if args.nCpu>0:
            import tensorflow as tf
            tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.nCpu))
            print('restrict CPU count to ',args.nCpu)

        dropFrac=args.dropFrac
        sh1=self.data['train']['X'].shape
        
        print('build_model inp1:',sh1,'design=',self.modelDesign)

        if self.modelDesign=='cnn1d': # . . . . . . . . . . . . . . . 
            xa = Input(shape=(sh1[1],),name='inp1d')
            h=Reshape((sh1[1],1))(xa)
            kernel = 5
            pool_len = 3 # how much time_bins get reduced per pooling
            cnnDim=[2,4]; numCnn=len(cnnDim)
            print(' cnn1Dim:',cnnDim)
            
            for i in range(numCnn):
                dim=cnnDim[i]
                h= Conv1D(dim,kernel,activation='relu', padding='valid',name='cnn%d_d%d_k%d'%(i,dim,kernel))(h)
                h= MaxPool1D(pool_length=pool_len, name='pool_%d'%(i))(h)
                print('cnn 1d',i,h.get_shape())

            h=Flatten(name='to_1d')(h)

        if self.modelDesign=='cnn2d': # . . . . . . . . . . . . . . . 
            xa = Input(shape=(sh1[1],sh1[2],),name='inp2d')      
            h=Reshape((sh1[1],sh1[2],1))(xa)
            kernel = 3
            pool_len = 2 # how much time_bins get reduced per pooling
            cnnDim=[4,8]; numCnn=len(cnnDim)
            print(' cnn2Dim:',cnnDim)
            
            for i in range(numCnn):
                dim=cnnDim[i]
                h= Conv2D(dim,kernel,activation='relu', padding='valid',name='cnn%d_d%d_k%d'%(i,dim,kernel))(h)
                h= MaxPool2D(pool_size=pool_len, name='pool_%d'%(i))(h)
                print('cnn 2d',i,h.get_shape())

            h=Flatten(name='to_1d')(h)

        if self.modelDesign=='lstm': # . . . . . . . . . . . . . . . 
            lstmDim=10
            recDropFrac=0.5*dropFrac
            print(' lstmDim:',lstmDim)
            h= LSTM(lstmDim, activation='tanh',recurrent_dropout=recDropFrac,dropout=dropFrac,name='lstmA_%d'%lstmDim,return_sequences=True) (h)
            h= LSTM(lstmDim, activation='tanh',recurrent_dropout=recDropFrac,dropout=dropFrac,name='lstmB_%d'%lstmDim,return_sequences=False) (h)

        print('pre FC=>',h.get_shape())
        h = Dropout(dropFrac,name='dropFC')(h)

        # .... FC  layers  COMMON 
        fcDim=[10,5]; numFC=len(fcDim)

        for i in range(numFC):
            dim = fcDim[i]
            h = Dense(dim,activation='relu',name='fc%d'%i)(h)
            h = Dropout(dropFrac,name='drop%d'%i)(h)
            print('fc',i,h.get_shape())

        y= Dense(1, activation='sigmoid',name='sigmoid')(h)

        lossName='binary_crossentropy'
        optimizerName='adam'
        
        print('build_model: loss=',lossName,' optName=',optimizerName,' out:',y.get_shape())
        # full model
        model = Model(inputs=xa, outputs=y)

        model.compile(optimizer=optimizerName, loss=lossName, metrics=['accuracy'])
        self.model=model
   
        model.summary() # will print
        print('model size=%.1fK compiled elaT=%.1f sec'%(model.count_params()/1000.,time.time() - start))
        


    #............................
    def train_model(self,args):
        X=self.data['train']['X']
        Y=self.data['train']['Y']
        X_val=self.data['val']['X']
        Y_val=self.data['val']['Y']
        print('train Xshape',X.shape)
        
        callbacks_list = []
        lrCb=MyLearningTracker()
        callbacks_list.append(lrCb)
        
        if args.earlyStopOn:
            earlyStop=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto',min_delta=0.001)
            callbacks_list.append(earlyStop)
            print('enabled EarlyStopping')

        if args.checkPtOn:
            #outFw='weights.{epoch:02d}-{val_loss:.2f}.h5'
            outF5w=self.prjName+'.weights_best.h5'
            ckpt=ModelCheckpoint(outF5w, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1,period=4)
            callbacks_list.append(ckpt)
            print('enabled ModelCheckpoint')

        if args.reduceLearn:
            redu_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.0, verbose=1,epsilon=0.01)
            callbacks_list.append(redu_lr)
            print('enabled ReduceLROnPlateau')

        print('\nTrain_model X:',X.shape, ' earlyStop=',args.earlyStopOn,' epochs=',args.epochs,' batch=',args.batch_size,)
        startTm = time.time()
        hir=self.model.fit(X,Y, callbacks=callbacks_list,
                           validation_data=(X_val,Y_val),
                           shuffle=True,
                           batch_size=args.batch_size, nb_epoch=args.epochs,
                           verbose=1)
        self.train_hirD=hir.history
        self.train_hirD['lr']=lrCb.hir
    

        #evaluate performance for the last epoch
        acc=self.train_hirD['val_acc'][-1]
        loss=self.train_hirD['val_loss'][-1]
        fitTime=time.time() - start
        print('\n End Validation Accuracy:%.3f'%acc, ', Loss:%.3f'%loss,', fit time=%.1f sec'%(fitTime))
        self.train_sec=fitTime


    #............................
    def make_prediction(self,dom):
        X=self.data[dom]['X']
        Yhot=self.data[dom]['Y']

        print('make_prediction, dom=',dom,' shape=',X.shape)
        Yprob = self.model.predict(X).flatten()
        print('Yprob',Yprob.shape)
        return X,Yhot,Yprob

 
    #............................
    def save_model_full(self):
        outF=self.outPath+'/'+self.prjName+'.'+self.modelDesign+'.model_full.h5'
        print('save model full to',outF)
        self.model.save(outF)
        xx=os.path.getsize(outF)/1048576
        print('closed  hdf5:',outF,' size=%.2f MB'%xx)
 
    #............................
    def load_model_full(self):
        try:
            del self.model
            print('delte old model')
        except:
            a=1
        start = time.time()
        outF5m=self.outPath+'/'+self.prjName+'.'+self.modelDesign+'.model_full.h5'
 
        print('load model and weights  from',outF5m,'  ... ')
        self.model=load_model(outF5m) # creates mode from HDF5
        self.model.summary()
        print(' model loaded, elaT=%.1f sec'%(time.time() - start))

    #............................
    def load_weights(self,name):
        start = time.time()
        outF5m=self.prjName+'.%s.h5'%name
        # print('load  weights  from',outF5m,end='... ')
        self.model.load_weights(outF5m) # creates mode from HDF5
        print('loaded, elaT=%.2f sec'%(time.time() - start))
        
