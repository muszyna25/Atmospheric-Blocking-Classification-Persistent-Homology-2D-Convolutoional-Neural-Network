import os
from netCDF4 import Dataset
import numpy as np
from sklearn import preprocessing
from ripser import ripser
from scipy.spatial.distance import pdist, cdist, squareform
import h5py
import math
from functools import reduce
from scipy.spatial import distance


class Persist_Homologyclass(object):
    def __init__(self, data_path, bin_mask_path, varname='t', norm='euclidean', maxdim=1):
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
    def generate_dataset(self):
        self.generate_list_of_files()
        self.generate_data_list()
        self.generate_labeled_data_list()

#............................
    def save_dataset_hdf5(self):
        self.save_dict_to_hdf5(self.l_2D_hist)

#............................
    def create_file_list(self):
        list_of_files = []
        for root, dirs, files in os.walk('.'):  # Walks through all files in the given directory.
            list_of_files = [filename for filename in files if filename.endswith(".nc")]
        file_list = sorted(list_of_files, key = lambda x: x.split('_')[1]) # Sort files for data format: 'name_year.nc'.
        return file_list

#............................
    def generate_list_of_files(self):
        # Climate netcdf data files.
        cwd = os.getcwd() 
        os.chdir(self.data_path)
        self.l_fns = self.create_file_list()         
        print(self.l_fns)
        print('[+] generate_list_of_files -- climate data -- done')
        
        # Binary mask netcdf files.
        cwd = os.getcwd() 
        os.chdir(self.bin_mask_path)
        self.l_lab_fns = self.create_file_list()         
        print(self.l_lab_fns)
        print('[+] generate_list_of_files -- binary masks -- done')
  
#............................
    def read_netcdf_file(self, path, fname, varname): #Variables names: e.g., 'lon', 'lat', 'prw'
        fh = Dataset(path + fname, mode='r')
        var_netcdf = fh.variables[varname][:] #Retrieves a given variable by name.
        return var_netcdf

#............................
    def read_netcdf_file_(self, path, fname, varname): #Variables names: e.g., 'lon', 'lat', 'prw'
        fh = Dataset(path + fname, mode='r')
        var_netcdf = fh.variables[varname][:] #Retrieves a given variable by name.
        #print(fh.dimensions)
        #print(fh.variables)
        return var_netcdf

#............................
    def preprocessing_norm_stand(self, img): #This function does normalization and standardization of the input data.
        img = preprocessing.normalize(img)
        #scaler = preprocessing.StandardScaler()
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        img = scaler.fit_transform(img)
        
        #X = img - np.mean(img,axis=0)
        #img = X/np.std(img, axis=0)
        return img

#............................
    def extract_subimages(self, img, nRegions): #nRegions means number of regions per hemisphere.
        lsubimgs = [];
        offsetX = 0 #This offset skips n-pixels in the vertical dimension.
        offsetY = 0 #This offset skips n-pixels in the horizontal dimension.
        nX = img.shape[0]; nY = img.shape[1]; 
        #print('Dimensions: ', nX, nY)
    
        #If number of regions is odd, which does not suit the science problem.
        if int(nRegions) % 2 != 0:
            nRegions = nRegions + 1 #Make it the even number.
            print('nRegions is not even number. Now it is even: ', nRegions)
    
        #Finds difference to adjust image size.
        if int(nX) % 2 != 0:
            tmp = np.floor(nX/2)
            offsetX = int(nX - tmp*2) #Calculates number of pixels to skip in the vertical dimension.
            img = img[:-offsetX, :]
            #print('OffsetX: ', offsetX)

        if int(nY) % nRegions != 0:
            tmp = np.floor(nY/nRegions)
            offsetY = int(nY - tmp*nRegions) #Calculates number of pixels to skip in the horizontal dimension.
            img = img[:, :-offsetY]
            #print('OffsetY: ', offsetY)
    
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
        #D = pdist(M, norm) #Computes all the pairwise distances.
        #SD = squareform(D) #Forms a matrix from the vector with pairwise distances.

        R = M.reshape((M.shape[0]*M.shape[1], 1))
        print('Reshaping')
        #SD = cdist(R, R, norm)
        #SD = np.tril(SD)
        # or
        PD = pdist(R, norm)
        print('pdist', PD.shape)
        SD = squareform(PD)
        
        #print(M[0][:])
        #lM = M.tolist()
        #llM=reduce(lambda x,y: x+y,lM)
        #print('List ', type(llM), len(llM))
        
        #print(llM)
        #M = np.array(M)
        #print('Original ', type(M), M.shape)
        #SD = cdist(np.asmatrix(M), np.asmatrix(M), norm)
        #SD = cdist(llM, llM, norm)
        
        #SD = np.array([])
        #for i in range(0, len(llM)):
        #    for j in range(0, len(llM)):
        #        SD=np.append(SD, llM[i]-llM[j])

        #SD = [np.abs(i[j]-i[j]) for j in range(0,len(i)) for i in lM]

        print('SD', type(SD), SD.shape)

        return SD

#............................
    def PH_func_call(self, data, norm, maxdim):
        X = self.compute_dist_mat(data, norm) #Computes distance matrix from a squared scalar field.
        #result = ripser(data, distance_matrix=True, maxdim=maxdim, metric='euclidean') #Ripser to compute persistent homology (it is C++ code with Python binding).
        result = ripser(X, distance_matrix=True, maxdim=maxdim, thresh=1.0) # Set threshold to speed up computations. 
        dgms = result['dgms']
        return dgms

#............................
    def hist_data(self, dgms): #Computes 2d histogram for dR = birth - death and mR = birth + death
        data = dgms[1][:]
        dR = np.array([np.around((x[1]-x[0])/2.0, decimals=8) for x in data]) #Should I take the mean?
        mR = np.array([np.around((x[1]+x[0])/2.0, decimals=8) for x in data])
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
        #self.outputData={dom:{'X': np.ndarray([]),'Y':np.array([])} for dom in ['train','val','test']}
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
    def load_hdf5_file(self, inputFile): # Load data to check if it was saved properly.
        h5f=h5py.File(inputFile, 'r')
        for k in h5f.keys():
            obj=h5f[k][:]
            print(obj.shape)
        h5f.close()

#............................
    def k_subimages_PH(self, I): 
        dgms = np.ndarray([])
        dgms = self.PH_func_call(I, self.norm, self.maxdim) #Computes H1 homologies.
        self.l_dgms.append(dgms)
        new_repres_img = self.hist_data(dgms) #Computes 2d histogram.
        return new_repres_img

#............................
    def generate_data_list(self):
        for i in range(1, len(self.l_fns)): ## change to 0.
            print('File:', self.l_fns[i])
            fd=self.read_netcdf_file(self.data_path, self.l_fns[i], self.varname)
            for j in range(110, len(fd)): 
                print('Timestep: %d' % j)
                self.l_global_imgs.append(fd[j])
                print('Global img size ', fd.shape)
                img = self.preprocessing_norm_stand(fd[j]) # Normalizes & standardizes data to get rid of noise.
                print('max img', np.max(img))
                imgs = self.extract_subimages(img, 4) # Extracts n images per hemisphere (here, 4*2 = 8 images in total).
                self.l_imgs.extend(imgs) # Extracts eight subimages.
                self.l_2D_hist.extend([self.k_subimages_PH(self.preprocessing_norm_stand(imgs[k])) for k in range(0, len(imgs))]) # Creates list of 2d histograms.
        print('[+] %i 2D hists: generate_data_list -- done' %len(self.l_2D_hist))

#............................
    def assign_label(self, l_imgs, threshold): #This function does not work well
        y = []
        th = threshold * int(l_imgs[0].shape[0]*l_imgs[0].shape[1])
        #print('Threshold: %i' %th)
        for i in range(0, len(l_imgs)):
            l = list(l_imgs[i].flatten())
            no_ones = l.count(1)
            #print('Threshold: %i vs number of ones: %i' %(th, no_ones))
            if no_ones >= th:
                y.extend([1])
            else:
                y.extend([0])
        return y

#............................
    def assign_label_v2(self, l_imgs, threshold):
        y = []
        n_imgs = len(l_imgs)
        x_size = l_imgs[0].shape[0]
        y_size = l_imgs[0].shape[1]
        mid_pt = [np.round(x_size/2.0), np.round(y_size/2.0)]

        th = 0.5
        x_start = int(mid_pt[0] - np.round(th*x_size))
        x_end = int(mid_pt[0] + np.round(th*x_size))
        y_start = int(mid_pt[1] - np.round(th*y_size))
        y_end = int(mid_pt[1] + np.round(th*y_size))

        for j in range(0, n_imgs):
            no_ones = list(l_imgs[j].flatten()).count(1)
            if no_ones == 0: continue
            sub_img = l_imgs[j][x_start:x_end][y_start:y_end] # Extract subimage.
            n_ones = list(sub_img.flatten()).count(1)
            t = threshold * sub_img.shape[0] * sub_img.shape[1]
            if n_ones >= t:
                y.extend([1])
            else:
                y.extend([0])
        return y

#............................
    def generate_labeled_data_list(self): 
        #print('generate_labeled_data_list')
        for i in range(1, len(self.l_lab_fns)): ## change to 0
            fd=self.read_netcdf_file(self.bin_mask_path, self.l_lab_fns[i], 'FLAG') # Variable name is fixed for these files.
            print('File:', self.l_lab_fns[i])
            for j in range(1458, len(fd)): #1457
                print('Timestep: %d' % j)
                img = fd[j]
                img = np.flip(img,0) # ETHZ guys saved the matrix fliped so it must be fliped by axis 0. 

                self.l_lab_global_imgs.append(img) # Store global binary images.

                l_subimages = self.extract_subimages(img, 4) # Extracts eight subimages.
                
                self.l_lab_imgs.extend(l_subimages)

                #l_y = self.assign_label(l_subimages, 0.01) # Start with 10% of pixels as ones (1's).

                l_y = self.assign_label_v2(l_subimages, 0.01)
                
                self.l_labels.extend(l_y)
        print('[+] generate_labeled_data_list -- done')

#............................
    def compute_deltaR_midR(self, root_degree, dgm, scale_flag):
        if scale_flag:
            dR = np.array([math.pow(np.around((x[1]-x[0])/2.0, decimals=8),1.0/root_degree) for x in dgm])  
            mR = np.array([math.pow(np.around((x[1]+x[0])/2.0, decimals=8),1.0/root_degree) for x in dgm])
        else:
            dR = np.array([np.around((x[1]-x[0])/2.0, decimals=8) for x in dgm])
            mR = np.array([np.around((x[1]+x[0])/2.0, decimals=8) for x in dgm])

        return (dR, mR)


