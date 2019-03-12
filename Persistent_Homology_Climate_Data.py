#!/usr/bin/env python
<<<<<<< HEAD
# coding: utf-8

# In[1]:

=======
>>>>>>> PH exists as separate class and as input for CNN.

from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
#matplotlib.use("Agg")
from scipy.spatial.distance import pdist, squareform
import h5py
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from matplotlib.ticker import FormatStrFormatter
import os
<<<<<<< HEAD
get_ipython().run_line_magic('matplotlib', 'inline')
=======
#get_ipython().run_line_magic('matplotlib', 'inline')
>>>>>>> PH exists as separate class and as input for CNN.
from sklearn import preprocessing
from persim import PersImage
from itertools import product
from scipy.stats import multivariate_normal as mvn
import random

<<<<<<< HEAD

# In[8]:


=======
>>>>>>> PH exists as separate class and as input for CNN.
def read_netcdf_file(path, fname, varname): #Variables names: e.g., 'lon', 'lat', 'prw'
    fh = Dataset(path + fname, mode='r')
    var_netcdf = fh.variables[varname][:] #Retrieves a given variable by name.
    print(var_netcdf.shape)
    return var_netcdf

def show_patch(M, figsizx, figsizy): #For example, y:[0:160] x:[0:320]
    plt.clf()
    plt.figure(figsize = (figsizx,figsizy))
    plt.imshow(M, interpolation='bilinear') #It does blinear interpolation of data to display image.
    plt.axis('off') #Turns off the ticks on both axises.
    plt.show()

def compute_dist_mat(M, norm): #e.g., norm: 'euclidean', etc.
    #print('Original ', M.shape)
    D = pdist(M, norm) #Computes all the pairwise distances.
    SD = squareform(D) #Forms a matrix from the vector with pairwise distances.
    return SD

def plot_barcode_and_pers_dgm(dgms, indx):
   
    for dim in range(0, len(dgms)):
        print('Homology:  ', dim)
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
        #plt.close(fig)
    
def adjust_numerical_stability(dmat):
    dmat += np.max(dmat) #Shifts the values to avoid numeric problems.
    #dmat -= np.max(dmat)
    return dmat

def create_file_list(st_offset):
    list_of_files = []
    for root, dirs, files in os.walk('.'):  #Walks through all files in the given directory.
        for filename in files:
            if filename.endswith(".nc"):
                list_of_files.append(filename)
    #file_list = sorted(list_of_files, key = lambda x: x.split('.')[0][st_offset:]) #Sort file list by extracted 'year' from the file name. Arg 'st_offset' must be adjusted to the file name format of specific datasets.
    file_list = sorted(list_of_files, key = lambda x: x.split('_')[1]) #For ERA-Interim data.
    return file_list

def generate_list_of_files(data_path):
    cwd = os.getcwd() 
    os.chdir(data_path)
    fn_list = create_file_list(6) #Creates file list and sorts it based on the extracted information from file name. Arg 'st_offset = 6' must be adjusted to the file name format of specific datasets.
    return fn_list

def PH_func_call(data, norm, maxdim):
    X = compute_dist_mat(data, norm) #Computes distance matrix from a squared scalar field.
    result = ripser(X, distance_matrix=True, maxdim=maxdim) #Calls Ripser to compute persistent homology (it is C++ code with Python binding).
    dgms = result['dgms']
    return dgms

def compute_bars_lengths(dgm):
    data = dgm[0:-1][:]
    print(data.shape)
    lbars = np.array([np.around(x[1]-x[0], decimals=8) for x in data])
    return lbars

def compute_histogram(dgms, indx):
    
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
        #plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10) #Sets a clean upper y-axis limit.
    fig.show()
    fig.savefig('Histogram_' + indx + '.png')
    #plt.close(fig)
    
def save_notebook_to_script():
    os.chdir('/global/homes/m/muszyng/project-cnn-tda/')
    get_ipython().system('jupyter nbconvert --to script Persistent_Homology_Climate_Data.ipynb #Converts jupyter notebook to python script.')
    os.getcwd()
    
def plot_multiple_imgs(l_imgs, indx):
    n = len(l_imgs); no_imgs = int(np.round(n/2))
    
    fig, axs = plt.subplots(nrows=2, ncols=no_imgs, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.01, right=0.07, hspace=0.01, wspace=0.01)
    
    for ax, img in zip(axs.flat, l_imgs): 
        ax.imshow(img, interpolation='bilinear', cmap='viridis') #Plots multiple subimages next to each other.
        
    plt.tight_layout() #Adjusts subplot params so that the subplots fit in to the figure area.
    plt.show()
    plt.savefig('Subimages_' + indx + '.png')
    #plt.close(fig)
    
'''Soulution to odd number of cols & rows--reduce image to square image.'''
def extract_subimages(img, nRegions): #nRegions means number of regions per hemisphere.
    lsubimgs = [];
    offsetX = 0 #This offset skips n-pixels in the vertical dimension.
    offsetY = 0 #This offset skips n-pixels in the horizontal dimension.
    nX = img.shape[0]; nY = img.shape[1]; print('Dimensions: ', nX, nY)
    
    #If number of regions is odd, which does not suit the science problem.
    if int(nRegions) % 2 != 0:
        nRegions = nRegions + 1 #Make it the even number.
        print('nRegions is not even number. Now it is even: ', nRegions)
    
    #Finds difference to adjust image size.
    if int(nX) % 2 != 0:
        tmp = np.floor(nX/2)
        offsetX = int(nX - tmp*2) #Calculates number of pixels to skip in the vertical dimension.
        img = img[:-offsetX, :]
        print('OffsetX: ', offsetX)

    if int(nY) % nRegions != 0:
        tmp = np.floor(nY/nRegions)
        offsetY = int(nY - tmp*nRegions) #Calculates number of pixels to skip in the horizontal dimension.
        img = img[:, :-offsetY]
        print('OffsetY: ', offsetY)
    
    #print('Square image size: ', img.shape)
    
    windowSizeX = int(img.shape[0]/2) # Subimage size in the vertical dimension.
    windowSizeY = int(img.shape[1]/nRegions) #Subimage size in the horizontal dimension.
    
    for x in range(0, img.shape[0], windowSizeX):
        for y in range(0, img.shape[1], windowSizeY):
            subimg = img[x:x + windowSizeX, y:y + windowSizeY]
            lsubimgs.append(subimg)
    return lsubimgs

'''This function is just for debugging'''
def create_synthetic_img(n,m): # n=240, m=480; Three horizontal lines dividing image into three horizontal sections.
    M = np.ones((n,m))
    M[60:70][0:] = 30
    M[119:130][0:] = 20
    M[180:190][0:] = 10
    return M

'''These two functions need to be improved'''
def save_to_hdf5(DM):
    hf = h5py.File('ar_vs_non_ar_data.h5', 'w')
    hf.create_dataset('dis_mat', data=DM) #Create dataset the same way as Jan did.
    hf.close()
    
def extract_small_patches(start_indx, varnetcdf, y1, y2, x1, x2):
    for i in range(start_indx, varnetcdf.shape[0]):
        M = varnetcdf[i][y1:y2, x1:x2]
    return M

def preprocessing_norm_stand(img): #This function does normalization and standardization of the input data.
    img = preprocessing.normalize(img)
    scaler = preprocessing.StandardScaler()
    img = scaler.fit_transform(img)
    return img


# ## Main Workflow

# ### Settings for dataset

# In[5]:


#ECMWF data.
ecmwf_data_path = '/global/cscratch1/sd/muszyng/ethz_data/ecmwf_data/' # Name of one file that I have: ECMWF_1979_Jan.nc

#ETHZ binary mask data.
ethz_data_path = '/global/cscratch1/sd/muszyng/ethz_data/iacftp.ethz.ch/pub_read/sprenger/grzegorz.muszynski/'

#Old CAM5(?) data.
prw_data_path = '/global/homes/m/muszyng/project-cnn-tda/ar_data/TECA_data/' # This the name of the file: prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231.nc

#Global path to data.
path = ecmwf_data_path

#Generate list of data files with climate model output.
fn_list = []; fn_list = generate_list_of_files(path); print(fn_list)

#Parameters for retrieving subimages from global image.
varname = ['pv', 'z', 't', 'u']; y1=0; y2=100; x1=0; x2=100; indx = 0; timestep = 70; var_netcdf = np.array([]); norm = 'cosine'; maxdim = 1;


# In[9]:


'''
f_plot = True # x_y_z: x -- a netcdf file; y -- a timestep (timeframe) from a file; z -- a subimage from a timestep
for i in range(0, len(fn_list)):
    fd = read_netcdf_file(path, fn_list[i], varname[0])
    for j in range(0, len(fd)):
        img = fd[j]
        img = preprocessing_norm_stand(img)
        if j == 3: break
        else:
            l_imgs = extract_subimages(img, 4)
            if f_plot: plot_multiple_imgs(l_imgs, str(i) + '_' + str(j))
            for k in range(0, len(l_imgs)):
                dgms = PH_func_call(l_imgs[k], norm, maxdim)
                if f_plot: plot_barcode_and_pers_dgm(dgms, str(i) + '_' + str(j) + '_' + str(k)); compute_histogram(dgms, str(i) + '_' + str(j) + '_' + str(k))
'''


# In[11]:


def hist_data(dgms): #Computes 2d histogram for dR = birth - death and mR = birth + death
    data = dgms[1]
    dR = np.array([np.around(x[1]-x[0], decimals=8) for x in data])
    mR = np.array([np.around((x[1]+x[0])/2, decimals=8) for x in data])
    #fig = plt.figure()
    nbin = np.linspace(0,0.5,28) #Here we set number of bins (2d cells) so in fact it sets up the size of image (e.g., from 0 to 0.5).
    #plt.hist2d(dR, mR, bins=nbin)
    counts, xedges, yedges = np.histogram2d(dR, mR, bins=nbin)
    #plt.imshow(counts)
    print(counts.shape, xedges.shape, yedges.shape)
    #plt.show()
    return counts
    
def add_rnd_noise(img): #Function creates fake second class by shuffling pixels of flatten image.
    xDim = img.shape[0]
    yDim = img.shape[1]
    print(img.shape)
    
    tmp_img = img.flatten() #Flattens an image.
    print(tmp_img.shape)
    
    np.random.shuffle(tmp_img) #Shuffles the flatten image (1d array).
    print(type(tmp_img))
    img = tmp_img.reshape(xDim, yDim) #Reshapes it back to 2d array.
    
    return img


# ### Debugging

# In[12]:


#Create three dictionaries according to Jan's convention (for training, validating, testing).
func2class_train = {"X": np.ndarray([]), "Y": np.array([])}
func2class_val = {"X": np.ndarray([]), "Y": np.array([])}
func2class_test = {"X": np.ndarray([]), "Y": np.array([])}

#Saves dictionaries in hd5 format.
def save_to_hdf5_(D, name): #D is dictionary.
    hf = h5py.File(name + '.hd5', 'w')
    
    for xy in D: #Loops over element in the dictionary.
        xobj=D[xy] #Check what this line does ????
        print('ss',xy,D[xy].shape)
        hf.create_dataset(xy, data=D[xy]) #Check what this line does too??
    hf.close()
        
    #xx=os.path.getsize(hf)/1048576
    #print('closed', 'size=%.2f MB'%xx)


# In[ ]:


list_new_data = []
for i in range(0, len(fn_list)):
    fd = read_netcdf_file(path, fn_list[i], varname[0])
    for j in range(0, len(fd)):
        img = fd[j] #Gets an image from a file.
        img = preprocessing_norm_stand(img) #Normalizes & standardizes data.
        l_imgs = extract_subimages(img, 4) #Extracts eight subimages.
        for k in range(0, len(l_imgs)):
            #if k % 2 == 0:
            #I = add_rnd_noise(l_imgs[k]) #Adds randomness to create the second class of objects in the input raw data.    
            dgms = PH_func_call(I, norm, maxdim) #Computes H1 homologies.
            new_repres_img = hist_data(dgms) #Computes 2d histogram.
            if k%2 == 0:
                np.random.shuffle(new_repres_img) #Adds randomness to every second 2d histogram.
            list_new_data.append(new_repres_img)


# In[ ]:


print(len(list_new_data)) #1888

func2class_train["X"] = np.array(list_new_data[0:1500]) # ~%80 training--2d histograms.
func2class_train["Y"] = np.array([a%2 for a in range(0, 1500)]) #Training set of labels.

func2class_val["X"] = np.array(list_new_data[1501:1700]) # ~10% validation set--2d histograms.
func2class_val["Y"] = np.array([a%2 for a in range(1501, 1700)]) #Validation set of labels.

func2class_test["X"] = np.array(list_new_data[1701:]) # ~10% testing set--2d histograms.
func2class_test["Y_"] = np.array([a%2 for a in range(1701, 1888)]) #Testing set of labels.

#Save all three dictionaries to seperate hdf5 format files.
save_to_hdf5_(func2class_train, '/global/cscratch1/sd/muszyng/ethz_data/ecmwf_data/tzfunc2class_train')
save_to_hdf5_(func2class_val, '/global/cscratch1/sd/muszyng/ethz_data/ecmwf_data/tzfunc2class_val')
save_to_hdf5_(func2class_test, '/global/cscratch1/sd/muszyng/ethz_data/ecmwf_data/tzfunc2class_test')


<<<<<<< HEAD
# In[ ]:
=======
# In[13]:
>>>>>>> PH exists as separate class and as input for CNN.


save_notebook_to_script()

