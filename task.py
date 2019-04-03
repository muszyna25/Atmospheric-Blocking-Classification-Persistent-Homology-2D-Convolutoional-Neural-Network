#!/usr/bin/env python

import os
from netCDF4 import Dataset
import numpy as np
from sklearn import preprocessing
from ripser import ripser
from scipy.spatial.distance import pdist, cdist, squareform
import h5py
import math
from scipy.spatial import distance
import sys
import time

class PH_class(object):
    def __init__(self, data_path='', bin_mask_path='', varname='t', norm='euclidean', maxdim=1):
        self.data_path=data_path
        self.bin_mask_path=bin_mask_path
        self.varname=varname

        # Params for peristent homology.
        self.norm = norm
        self.maxdim = maxdim

        # Lists for data.
        self.l_fns = []
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
    def compute_dist_mat(self, M, norm): #e.g., norm: 'euclidean', etc.
        print('Original ', type(M), M.shape)
        # Row pairwise distance.
        R = M.reshape((M.shape[0]*M.shape[1], 1))
        print('Reshaping', R.shape)
        PD = pdist(R, norm)
        print('pdist', PD.shape)
        SD = squareform(PD)
        SD = np.tril(SD)
        print('SD', type(SD), SD.shape)
        return SD

#............................
    def PH_func_call(self, data, norm, maxdim):
        X = self.compute_dist_mat(data, norm) #Computes distance matrix from a squared scalar field.
        start = time.time()
        result = ripser(X, distance_matrix=True, maxdim=maxdim, thresh=1.0) # Set threshold to speed up computations. 
        end = time.time()
        print('ripser time %i' %(end-start))
        dgms = result['dgms']
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
    def save_dict_to_hdf5_(self, dataList, input_file, file_part, frame_id):
        sizeTrain=0.7
        sizeVal=0.2
        sizeTest=0.1
        nImgs=len(dataList)
        nTrain=round(nImgs*sizeTrain)
        nVal=round(nImgs*sizeVal)
        nTest=round(nImgs*sizeTest)
        #hf=h5py.File(input_file + '_' + file_part + '_' + frame_id + '.hd5', 'w')
        #hf.create_dataset(xy, data=self.outputData[ktvt][xy]) #ktvt is the first key, xy is the key as dictionary.
        #hf.close()
        #print('[+] save_dict_to_hdf5 -- done')

#............................
    def k_subimages_PH(self, I): 
        dgms = np.ndarray([])
        dgms = self.PH_func_call(I, self.norm, self.maxdim) #Computes H1 homologies.
        self.l_dgms.append(dgms)
        new_repres_img = self.hist_data(dgms) #Computes 2d histogram.
        print('histogram size: ', len(new_repres_img))
        return new_repres_img

#............................
    def generate_data_list(self, idx, sub_img_id):
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
    def save_dataset_hdf5(self, input_file, file_part, frame_id):
        self.save_dict_to_hdf5_(self.l_2D_hist, input_file, file_part, frame_id)

if __name__ == "__main__":

    input_file = sys.argv[1]
    file_part = int(sys.argv[2])
    sub_img_id = int(sys.argv[3])
    proc_id = 0 #int(os.environ['SLURM_PROCID'])
    frame_id = int(proc_id + file_part*62)

    print(file_part, frame_id, sub_img_id)

    ph = PH_class() # Optionals: 1) var name; 2) metric pairwise dist matrix; 3) max homology group dim.

    # Generate dataset: (X - features, Y - labels).
    ph.l_fns.append(input_file)
    ph.generate_data_list(frame_id, sub_img_id)
    print('Task complete %i' %proc_id)

    #ph.save_dataset_hdf5_(input_file, file_part, frame_id) # Save dataset to hdf5 format: (train, val, test).

