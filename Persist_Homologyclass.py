import os
from netCDF4 import Dataset
import numpy as np
from sklearn import preprocessing
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import h5py

class Persist_Homologyclass(object):
    def __init__(self, datapath, varname, norm='cosine', maxdim=1):
        self.datapath=datapath
        self.varname=varname
        self.norm = norm
        self.maxdim = maxdim
        self.fn_list = []
        self.list_new_data = []
        self.outputData={dom:{'X': np.ndarray([]),'Y':np.array([])} for dom in ['train','val','test']}
        
    def create_file_list(self, st_offset):
        list_of_files = []
        for root, dirs, files in os.walk('.'):  # Walks through all files in the given directory.
            for filename in files:
                if filename.endswith(".nc"): # If a netCDF file.
                    list_of_files.append(filename)
                    # file_list = sorted(list_of_files, key = lambda x: x.split('.')[0][st_offset:]) 
                    # Sort file list by extracted 'year' from the file name. 
                    # E.g., arg 'st_offset' must be adjusted to the file name format of specific datasets.
        file_list = sorted(list_of_files, key = lambda x: x.split('_')[1]) # For ERA-Interim data.
        return file_list

    def generate_list_of_files(self):
        cwd = os.getcwd() 
        os.chdir(self.datapath)
        # Creates file list and sorts it based on the extracted information from file name. 
        # E.g., arg 'st_offset = 6' must be adjusted to the file name format of specific datasets.
        self.fn_list = self.create_file_list(6)         
        print(self.fn_list)
  
    def read_netcdf_file(self, path, fname, varname): #Variables names: e.g., 'lon', 'lat', 'prw'
        fh = Dataset(path + fname, mode='r')
        var_netcdf = fh.variables[varname][:] #Retrieves a given variable by name.
        #print(fname, var_netcdf.shape)
        return var_netcdf

    def preprocessing_norm_stand(self, img): #This function does normalization and standardization of the input data.
        img = preprocessing.normalize(img)
        scaler = preprocessing.StandardScaler()
        img = scaler.fit_transform(img)
        return img

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
    
    def compute_dist_mat(self, M, norm): #e.g., norm: 'euclidean', etc.
        #print('Original ', M.shape)
        D = pdist(M, norm) #Computes all the pairwise distances.
        SD = squareform(D) #Forms a matrix from the vector with pairwise distances.
        return SD

    def PH_func_call(self, data, norm, maxdim):
        X = self.compute_dist_mat(data, norm) #Computes distance matrix from a squared scalar field.
        result = ripser(X, distance_matrix=True, maxdim=maxdim) #Calls Ripser to compute persistent homology (it is C++ code with Python binding).
        dgms = result['dgms']
        return dgms

    def hist_data(self, dgms): #Computes 2d histogram for dR = birth - death and mR = birth + death
        data = dgms[1]
        dR = np.array([np.around(x[1]-x[0], decimals=8) for x in data])
        mR = np.array([np.around((x[1]+x[0])/2, decimals=8) for x in data])
        nbin = np.linspace(0,0.5,28) #Here we set number of bins (2d cells) so in fact it sets up the size of image (e.g., from 0 to 0.5).
        counts, xedges, yedges = np.histogram2d(dR, mR, bins=nbin)
        #print(counts.shape, xedges.shape, yedges.shape)
        return counts

    #Saves dictionaries in hd5 format.
    def save_to_hdf5_(self, D, name): #D is dictionary.
        hf = h5py.File(name + '.hd5', 'w')
        for xy in D: #Loops over element in the dictionary.
            print('ss',xy,D[xy].shape)
            hf.create_dataset(xy, data=D[xy]) #Check what this line does too??
        hf.close()

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

    def load_hdf5_file(self, inputFile): # Load data to check if it was saved properly.
        h5f=h5py.File(inputFile, 'r')
        for k in h5f.keys():
            obj=h5f[k][:]
            print(obj.shape)
        h5f.close()

    def generate_data_list(self):
        print('generate_data_list')
        for i in range(1, len(self.fn_list)):
            fd=self.read_netcdf_file(self.datapath, self.fn_list[i], self.varname)
            print('File:', self.fn_list[i])
            for j in range(0, len(fd)):
                print('Timestep: %d' % j)
                img = fd[j] #Gets an image from a file.
                img = self.preprocessing_norm_stand(img) #Normalizes & standardizes data.
                l_imgs = self.extract_subimages(img, 4) #Extracts eight subimages.
                for k in range(0, len(l_imgs)):
                    #if k % 2 == 0: #I = add_rnd_noise(l_imgs[k]) #Adds randomness to create the second class of objects in the input raw data.    
                    I = l_imgs[k]
                    dgms = self.PH_func_call(I, self.norm, self.maxdim) #Computes H1 homologies.
                    new_repres_img = self.hist_data(dgms) #Computes 2d histogram.
                    if k%2 == 0:
                        np.random.shuffle(new_repres_img) #Adds randomness to every second 2d histogram.
                    self.list_new_data.append(new_repres_img)


