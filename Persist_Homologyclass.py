import os
from netCDF4 import Dataset
import numpy as np
from sklearn import preprocessing
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import h5py

class Persist_Homologyclass(object):
    def __init__(self, dimension):
        self.dimension = dimension
        
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

    def generate_list_of_files(self, data_path):
        cwd = os.getcwd() 
        os.chdir(data_path)
        # Creates file list and sorts it based on the extracted information from file name. 
        # E.g., arg 'st_offset = 6' must be adjusted to the file name format of specific datasets.
        fn_list = self.create_file_list(6)         
        return fn_list
    
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
            xobj=D[xy] #Check what this line does ????
            print('ss',xy,D[xy].shape)
            hf.create_dataset(xy, data=D[xy]) #Check what this line does too??
        hf.close()

