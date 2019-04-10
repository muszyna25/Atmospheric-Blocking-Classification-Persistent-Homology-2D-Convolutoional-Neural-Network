#!/usr/bin/env python

import os
from netCDF4 import Dataset
import numpy as np
from sklearn import preprocessing
from ripser import ripser
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import distance_matrix
import h5py
import math
from scipy.spatial import distance
import sys
import time

class PH_class(object):
    def __init__(self, data_path='', bin_mask_path='', varname='t', norm='cosine', maxdim=1):
        self.data_path=data_path
        self.bin_mask_path=bin_mask_path
        self.varname=varname

        # Params for peristent homology.
        self.norm = norm
        self.maxdim = maxdim

        # Lists for data.
        self.l_fns = []
        self.l_1D_hist = []
        self.l_2D_hist = []
        self.l_imgs = []
        self.l_dgms = []
        self.l_global_imgs = []

        # Lists for binary masks.
        self.l_lab_imgs = []
        self.l_lab_fns = []         
        self.l_lab_global_imgs = []
        self.l_labels = []
        
        self.outputData={dom:{'X': np.ndarray([]),'Y':np.array([])} for dom in ['train','val','test']} # Output for hdf5 format.
  
#............................
    def read_netcdf_file(self, fname, varname): #Variables names: e.g., 'lon', 'lat', 'prw'
        fh = Dataset(fname, mode='r')
        var_netcdf = fh.variables[varname][:] #Retrieves a given variable by name.
        fh.close()
        return var_netcdf

#............................
    def preprocessing_norm_stand(self, img): #This function does normalization and standardization of the input data.
        img = preprocessing.normalize(img)
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        img = scaler.fit_transform(img)
        return img

#............................
    def extract_subimages(self, img, nRegions): #nRegions means number of regions per hemisphere.
        lsubimgs = [];
        offsetX = 0 #This offset skips n-pixels in the vertical dimension.
        offsetY = 0 #This offset skips n-pixels in the horizontal dimension.
        nX = img.shape[0]; nY = img.shape[1]; 
    
        #If number of regions is odd, which does not suit the science problem.
        if int(nRegions) % 2 != 0:
            nRegions = nRegions + 1 #Make it the even number.
            print('nRegions is not even number. Now it is even: ', nRegions)
    
        #Finds difference to adjust image size.
        if int(nX) % 2 != 0:
            tmp = np.floor(nX/2)
            offsetX = int(nX - tmp*2) #Calculates number of pixels to skip in the vertical dimension.
            img = img[:-offsetX, :]

        if int(nY) % nRegions != 0:
            tmp = np.floor(nY/nRegions)
            offsetY = int(nY - tmp*nRegions) #Calculates number of pixels to skip in the horizontal dimension.
            img = img[:, :-offsetY]
    
        windowSizeX = int(img.shape[0]/2) # Subimage size in the vertical dimension.
        windowSizeY = int(img.shape[1]/nRegions) #Subimage size in the horizontal dimension.
    
        for x in range(0, img.shape[0], windowSizeX):
            for y in range(0, img.shape[1], windowSizeY):
                subimg = img[x:x + windowSizeX, y:y + windowSizeY]
                lsubimgs.append(subimg)
        
        return lsubimgs
    
#............................
    def pw_dist(self, M):
        R = M.reshape((M.shape[0]*M.shape[1], 1))
        return R
#............................
    def compute_dist_mat(self, M, norm): #e.g., norm: 'euclidean', etc.
        print('Original ', type(M), M.shape)
        #D = pdist(M, norm) #Computes all the pairwise distances.
        #SD = squareform(D) #Forms a matrix from the vector with pairwise distances.
        ''' 
        # Row pairwise distance.
        R = M.reshape((M.shape[0]*M.shape[1], 1))
        print('Reshaping', R.shape)
        PD = pdist(R, 'euclidean')
        #PD = cdist(R,R, 'euclidean')
        print('pdist', PD.shape)
        SD = squareform(PD)
        #SD = np.tril(SD) # Take lower triangular matrix.
        SD = np.triu(SD) # Take lower triangular matrix.
        print('SD', type(SD), SD.shape)
        '''
        '''
        R = M.reshape((M.shape[0]*M.shape[1], 1))
        SD = distance_matrix(R,R,2)
        '''

        R = M.reshape((M.shape[0]*M.shape[1], 1))
        print('Reshaping', R.shape)
        PD = pdist(R, 'euclidean')
        #PD = pdist(R, 'cityblock')

        #PD = pdist(R, lambda u, v: abs(u-v))
        #PD = pdist(R, lambda u, v: min(u,v))

        SD = squareform(PD, 'tomatrix')
        print('SD', type(SD), SD.shape)
        return SD

#............................
    def PH_func_call(self, data, norm, maxdim):
        X = self.compute_dist_mat(data, norm) #Computes distance matrix from a squared scalar field.
        #print('X ', np.argwhere(np.isnan(X)))
        start = time.time()
        #result = ripser(X, distance_matrix=True, maxdim=maxdim, thresh=1.0) # Set threshold to speed up computations. 
        result = []
        #result = ripser(X, distance_matrix=True, maxdim=1) # Set threshold to speed up computations. 
        result = ripser(X, distance_matrix=True, maxdim=1, thresh=1.0) # Set threshold to speed up computations. 
        end = time.time()
        print('ripser time %i' %(end-start))
        dgms = result['dgms']
        print('dgms ', dgms)
        return dgms

#............................
    def hist_data(self, dgms): #Computes 2d histogram for dR = birth - death and mR = birth + death
        data = dgms[1][:]
        #dR = np.array([np.around((x[1]-x[0])/2.0, decimals=8) for x in data]) 
        #mR = np.array([np.around((x[1]+x[0])/2.0, decimals=8) for x in data])
        root_degree=3
        dR = np.array([math.pow(np.around((x[1]-x[0])/2.0, decimals=8),1.0/root_degree) for x in data])  
        mR = np.array([math.pow(np.around((x[1]+x[0])/2.0, decimals=8),1.0/root_degree) for x in data])
        nbin = np.linspace(0,0.5,28) #Here we set number of bins (2d cells) so in fact it sets up the size of image (e.g., from 0 to 0.5).
        counts, xedges, yedges = np.histogram2d(dR, mR, bins=nbin)
        return counts

#............................
    def one_dim_hist_data(self, dgms):
        data = dgms[0][:-1]
        nbins = np.linspace(0,0.5,28)
        hist, bin_edges = np.histogram(data, nbins)
        return hist

#............................
    #Saves dictionaries in hd5 format.
    def save_to_hdf5_(self, D, name): #D is dictionary.
        hf = h5py.File(name + '.hd5', 'w')
        for xy in D: #Loops over element in the dictionary.
            print('ss',xy,D[xy].shape)
            hf.create_dataset(xy, data=D[xy]) #Check what this line does too??
        hf.close()

#............................
    def save_dict_to_hdf5(self, dataList):
        sizeTrain=0.7
        sizeVal=0.2
        sizeTest=0.1
        nImgs=len(dataList)
        nTrain=round(nImgs*sizeTrain)
        nVal=round(nImgs*sizeVal)
        nTest=round(nImgs*sizeTest)
        
        self.outputData['train']["X"] = np.array(dataList[0:nTrain]) # ~%80 training--2d histograms.
        self.outputData['train']["Y"] = np.array([a%2 for a in range(0, nTrain)]) #Training set of labels.

        self.outputData['val']["X"] = np.array(dataList[nTrain:nTrain+nVal]) # ~10% validation set--2d histograms.
        self.outputData['val']["Y"] = np.array([a%2 for a in range(0, nTrain+nVal)]) #Validation set of labels.

        self.outputData['test']["X"] = np.array(dataList[nTrain+nVal:]) # ~10% testing set--2d histograms.
        self.outputData['test']["Y"] = np.array([a%2 for a in range(0, nTrain+nVal+nTest)]) #Testing set of labels.

        for ktvt in self.outputData:
            hf=h5py.File(ktvt + '.hd5', 'w')
            for xy in self.outputData[ktvt]:
                hf.create_dataset(xy, data=self.outputData[ktvt][xy]) #ktvt is the first key, xy is the key as dictionary.
            hf.close()
        print('[+] save_dict_to_hdf5 -- done')

#............................
    def save_dataset_hdf5(self, input_file, file_part, frame_id):
        self.save_dict_to_hdf5_(self.l_2D_hist, input_file, file_part, frame_id)

#............................
    def save_barcodes(self, file_idx, idx):
        for dgm in self.l_dgms:
            print('Diagram ', dgm)
            dgm_H_0 = dgm[0][:-1]
            dgm_H_1 = dgm[1][:]
            print('H0 ', dgm_H_0)
            #USE np.savez()
            np.savez('PH0_' + str(file_idx) + '_' + str(idx) + '.npz', dgm_H_0)
            np.savez('PH1_' + str(file_idx) + '_' + str(idx) + '.npz', dgm_H_1)
        
        print('[+] save_barcodes -- done')

#............................
    def save_histograms(self, file_idx, idx):
        for i in range(0, len(self.l_2D_hist)):
            h1 = self.l_1D_hist[i] 
            h2 = self.l_2D_hist[i]
            #USE np.savez()
            np.savez('H1_' + str(file_idx) + '_' + str(idx) + '.npz', h1)
            np.savez('H2_' + str(file_idx) + '_' + str(idx) + '.npz', h2)
        
        print('[+] save_histograms -- done')

#............................
    def generate_data_list_old(self, idx, sub_img_id):
        print('File:', self.l_fns[0])
        fd=self.read_netcdf_file(self.l_fns[0], self.varname)
        print('Timestep: %d' % idx)
        self.l_global_imgs.append(fd[idx])
        print('Global img size ', fd.shape)
        print('fd[idx] ', fd[idx][0].shape)
        img = self.preprocessing_norm_stand(fd[idx][0]) # 0 corresponds to the first pressure level.
        print('max img', np.max(img))
        Imgs = self.extract_subimages(img, 4) # Extracts n images per hemisphere (here, 4*2 = 8 images in total).

        imgs = Imgs[sub_img_id]

        self.l_imgs.extend(imgs) # Extracts eight subimages.
        self.l_2D_hist.extend(self.k_subimages_PH(imgs)) # Creates list of 2d hist.
        print('[+] %i 2D hists: generate_data_list -- done' %len(self.l_2D_hist))

#............................
    def generate_data_list__oldv2(self, idx):
        for i in range(0, len(self.l_fns)): 
            print('File:', self.l_fns[i])
            fd=self.read_netcdf_file(self.l_fns[i], self.varname)
            #frame = fd[idx][0]
            frame = fd[idx]
            print('Global img shape: ', frame.shape)
            print('Timestep: %d' % idx)
            self.l_global_imgs.append(frame)
            img = self.preprocessing_norm_stand(frame) # Normalizes & standardizes data to get rid of noise.
            imgs = self.extract_subimages(img, 4) # Extracts n images per hemisphere (here, 4*2 = 8 images in total).
            self.l_imgs.extend(imgs) # Extracts 8 subimages.
            self.l_2D_hist.extend([self.k_subimages_PH(self.preprocessing_norm_stand(imgs[k])) for k in range(0, len(imgs))]) 
        print('[+] %i 2D hists: generate_data_list -- done' %len(self.l_2D_hist))

#............................
    def k_subimages_PH(self, I): 
        #dgms = np.ndarray([])
        #print('I ', I)
        dgms = self.PH_func_call(I, self.norm, self.maxdim) #Computes H1 homologies.
        print('Ripser output: ', dgms[1][:])
        self.l_dgms.append(dgms)
        
        hist_1D = self.one_dim_hist_data(dgms)
        hist_2D = self.hist_data(dgms) #Computes 2d histogram.

        return hist_1D, hist_2D

#............................
    def generate_data_list(self, idx):
        print('File:', self.l_fns)
        data = np.load(self.l_fns[0])
        fd = data.files
        #for i in fd[idx]:   # CHANGE THIS LINE IF DEBUGGING.
        #frame = data[i]
        frame = data[fd[idx]]
        print('Global img shape: ', frame.shape)
        print('Timestep: %s' % fd[idx])
        self.l_global_imgs.append(frame)
        img = self.preprocessing_norm_stand(frame) # Normalizes & standardizes data to get rid of noise.
        imgs = self.extract_subimages(img, 4) # Extracts n images per hemisphere (here, 4*2 = 8 images in total).
        #print('imgs ', imgs[0])
        self.l_imgs.extend(imgs) # Extracts 8 subimages.
        
        #all_histograms = [self.k_subimages_PH(imgs[k]) for k in range(0, len(imgs))]
        all_histograms = [self.k_subimages_PH(self.preprocessing_norm_stand(imgs[k])) for k in range(0, len(imgs))]
        for hist in all_histograms:
            self.l_1D_hist.extend(hist[0]) 
            self.l_2D_hist.extend(hist[1]) 
            
        print('[+] 1D & 2D hists: generate_data_list -- done')

if __name__ == "__main__":

    proc_id = 0#int(os.environ['SLURM_PROCID']) # CHANGE IT TO ZERO FOR TESTING/DEBUGGING.

    # Preprocess input args.
    input_file = sys.argv[1]
    file_idx = input_file.split('_')[3]
    print('[+] ====== ' + sys.argv[0] + ' Input file: ' + input_file + ' File index: ' + file_idx + ' Process index: ' + str(proc_id) + ' ======')

    ph = PH_class() # Optionals: 1) var name; 2) metric pairwise dist matrix; 3) max homology group dim.

    # Generate dataset: (X - features, Y - labels).
    #ph.l_fns.append(input_file)
    ph.l_fns.append('/global/cscratch1/sd/muszyng/ethz_data/ecmwf_data/ECMWF_1979_Jan.nc')
    ph.generate_data_list__oldv2(proc_id)
    #ph.generate_data_list(proc_id)

    # Save files to .npz format. 
    ph.save_barcodes(file_idx, proc_id)
    ph.save_histograms(file_idx, proc_id)
    print('[*] ====== Task complete for file %s ======' %(input_file))

