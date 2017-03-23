import dicom
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import dicom 
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import time
import pickle
import random
import h5py
import operator
from scipy import signal

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#------------------------------------------------------------------------#
#               load scan and calculate slice thickness                  #
#------------------------------------------------------------------------#
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        print(slice_thickness)
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


#------------------------------------------------------------------------#
#           convert from stored values to Hounsfield Unit (HU)           #
#------------------------------------------------------------------------#
def convert_pix2hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    
    # Set outside-of-scan pixels to 0, i.e., the air
    outside_image = image.min()
    image[image == outside_image] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        # https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
        # hu = slope*stored_value + intercept
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
        
    return np.array(image, dtype=np.int16)


#------------------------------------------------------------------------#
#                       plot the subsampled slices                       #
#------------------------------------------------------------------------#
def plot_subsampled_slices(slices, T = 5):
    # plot 1 slice every T slices, 4 slices per row
    # all images are scaled to plt.cm.bone color map: 
    # More info: http://matplotlib.org/examples/color/colormaps_reference.html
    f, plots = plt.subplots(int(slices.shape[0] / (4*T)) + 1, 4, figsize=(25, 25))
    for i in range(0, slices.shape[0], T):
        plots[int(i / (4*T)), int((i % (4*T)) / T)].axis('off')
        plots[int(i / (4*T)), int((i % (4*T)) / T)].imshow(slices[i], cmap=plt.cm.bone)

    
#------------------------------------------------------------------------#
#           resampling the slices (tensor) by using interpolation        #
#INPUT:                                                                  #
#   image: all slices in hu values                                       #
#   scan: the original slices containing thickness infor                 #
#   new_spacing: the expected spacing                                    #
#OUTPUT:                                                                 #
#   the resampled slices                                                 #
#------------------------------------------------------------------------#
def resample(image, scan, new_spacing=[1.0, 1.0, 1.0]):    
    # Determine current pixel spacing
    current_spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    resize_factor = current_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image
    

#------------------------------------------------------------------------#
#                   calculate gaussian distribution                      #
#------------------------------------------------------------------------#
def gauss(x, mu, sigma):
    return (1/np.sqrt(2*sigma*sigma*np.pi))*np.exp(-(x-mu)**2/(2.*sigma**2))

#------------------------------------------------------------------------#
#           perform optimization to find optimal mean and std            #
#INPUT:                                                                  #
#   INPUT_FOLDER: the folder containing *.hdf5 files                     #
#   OUTPUT_FOLDER: the folder containing the output files                #
#   loadHX: 1 if the normalized histogram (H) and its bin edges can be   #
#   loaded, 0 if not                                                     #
#------------------------------------------------------------------------#
def best_slice_cut(INPUT_FOLDER, OUTPUT_FOLDER, loadHX = 1):
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    if '.DS_Store' in patients:
        patients.remove('.DS_Store')
    
    npatients = len(patients)
    nbins = 150 
    # maximum number of bins
    length = 500 # max [x, y, z] = [490, 490, 428]

    H = np.zeros((npatients, length))
    X = np.zeros((npatients, length))
    S = np.zeros(npatients)
    lung_range = [-1000, -500]
    if loadHX == 0:
        for i,p in enumerate(patients):
            print('\nProcessing patient: {0}\n'.format(p))
            f = h5py.File(INPUT_FOLDER + "/" + p,  "r")
            pix_resampled = f['data']
            # number of pixels belonging to lungs
            lung_pixels = [(np.multiply(s>=lung_range[0], s<=lung_range[1])).sum() for s in pix_resampled]
            x = scipy.arange(len(lung_pixels))
            y = lung_pixels

            H[i,:len(y)] = y/np.sum(y)
            X[i,:] = np.array(range(length))
            S[i] = len(y)

        print('save H, X, S')
        pickle.dump([H, X, S], open(OUTPUT_FOLDER + '/' + 'H_X_S.p', 'wb'))
    else:
        [H, X, S] = pickle.load(open(OUTPUT_FOLDER + '/' + 'H_X_S.p', 'rb'))

    Mu = np.random.rand(npatients, 1)
    std = random.random()
    print('\nCheck gradient... {0}\n'.format(check_grad(Mu, X, H, std)))

    # initializations of std and Mu
    std = np.mean(np.std(X, axis=1))
    Mu = np.ones((X.shape[0],1)) * nbins / 2

    # find optimal std, mu
    opt_Mu, opt_sigma = grad_descent(Mu, X, H, std, eta = 50.0, max_iters = 20000)
    print('Init sigma = {0} --> opt sigma = {1}'.format(std, opt_sigma))
    
    print('save optimal Mu and sigma')
    pickle.dump([opt_Mu, opt_sigma], open(OUTPUT_FOLDER + '/' + 'opt_Mu_sigma.p', 'wb'))

    # for checking results
    # p = 0
    # while p < 20:
    #     hist_fit = gauss(X[p,], Mu[p,], opt_sigma)
    #     plt.plot(X[p,], H[p,], label='Test data')
    #     plt.plot(X[p,], hist_fit, label='Fitted data')
    #     hist_fit = gauss(X[p,], opt_Mu[p,], opt_sigma)
    #     plt.plot(X[p,], hist_fit, label='Fitted data')
    #     plt.show()
    #     p += 1

    # find good cutoff value
    cutoff = 0.9
    finds = np.zeros((X.shape[0], 3), dtype=int)
    stop = False
    while not stop: 
        stop = True
        for i in range(X.shape[0]):
            a,b = scipy.stats.norm(opt_Mu[i,], opt_sigma).interval(cutoff)  
            if a < 0:
                b = b + np.abs(a)
                a = 0
            if b > S[i]:
                cutoff -= 0.01
                stop = False
                print('a = {0} and b = {1}, width = {2}'.format(a, b, b - a))
                break
            finds[i,:] = [a, b, b - a]

    # trick for dealing with rounding error
    finds[:,2] = np.min(finds[:,2])
    finds[:,1] = finds[:,0] + finds[:,2]

    print('stop at cutoff: ', cutoff)
    pickle.dump(finds, open(OUTPUT_FOLDER + '/' + 'finds.p', 'wb')) 



#-------------------------------------------------------------------------------------------#
#                calculate the gradient of at Mu and sigma                                  #
#   J(Mu, sigma) = 0.5 * sum_i sum_j ((exp(-(Mu_i-x_ij)^2 / 2*sigma^2)/sqrt(2*sigma^2*pi))  #
#                  - H_ij)^2                                                                #
#-------------------------------------------------------------------------------------------#
def grad(Mu, X, H, sigma):
    n,d = X.shape # X.shape = H.shape
    tmp = X - Mu
    g = np.zeros((n+1,1))
    term1 = (1/np.sqrt(2*sigma*sigma*np.pi))*np.exp(-tmp*tmp/(2*sigma*sigma)) - H
    term2 = (tmp/(sigma*sigma))*(np.exp(-tmp*tmp/(2*sigma*sigma))/np.sqrt(2*sigma*sigma*np.pi))
    g[:n,] = np.sum(term1 * term2, axis=1).reshape(n,1)
    
    term3 = np.exp((-tmp*tmp)/(2*sigma*sigma))
    term3 = term3 * (-np.sqrt(2*np.pi)/(2*sigma*sigma*np.pi) - ((tmp*tmp)/((sigma**3)*np.sqrt(2*sigma*sigma*np.pi))))
    g[n,] = np.sum(term1 * term3)

    return g

#-------------------------------------------------------------------------------------------#
#                                       calculate the cost                                  #
#   J(Mu, sigma) = 0.5 * sum_i sum_j ((exp(-(Mu_i-x_ij)^2 / 2*sigma^2)/sqrt(2*sigma^2*pi))  #
#                  - H_ij)^2                                                                #
#-------------------------------------------------------------------------------------------#
def cost(Mu, X, H, sigma):
    n, d = X.shape # X.shape = H.shape
    tmp = X - Mu
    c = (1/np.sqrt(2*sigma*sigma*np.pi))*np.exp(-tmp*tmp/(2*sigma*sigma)) - H
    c = 0.5*c*c
    return np.sum(c)

#-------------------------------------------------------------------------------------------#
#               check derivative calculation by using numeric gradient                      #
#-------------------------------------------------------------------------------------------#
def check_grad(Mu, X, H, sigma):
    grad1 = grad(Mu, X, H, sigma)
    grad2 = numerical_grad(Mu, X, H, sigma)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

def numerical_grad(Mu, X, H, sigma):
    eps = 1e-4
    sigma_p = sigma + eps
    sigma_n = sigma - eps
    g = (cost(Mu, X, H, sigma_p) - cost(Mu, X, H, sigma_n))/(2*eps)
    return g

#-------------------------------------------------------------------------------------------#
#               perform gradient descent to find optimal Mu and sigma                       #
#INPUT:                                                                                     #
#   Mu_init: initialization values of Mu, Mu_int.shape = (n,1)                              #
#   sigma_init: initialization value of sigma                                               #
#   eta: learning rate                                                                      #
#   max_iters: the number of maximum iterations                                             #
#OUTPUT:                                                                                    #
#   opt_Mu: optimal value of Mu                                                             #
#   opt_sigma: optimal value of sigma                                                       #
#-------------------------------------------------------------------------------------------#
def grad_descent(Mu_init, X, H, sigma_init, eta = 1.0, max_iters = 100000):
    n,d = X.shape # X.shape = H.shape
    Mu = Mu_init.copy()
    sigma = sigma_init
    g = grad(Mu, X, H, sigma)
    print('\nIter \t Gradient \t Cost \n ')
    v = 0
    for it in range(max_iters): 
        v_new = 0.9*v + eta*g
        # update sigma   
        Mu_new = Mu - v_new[:n,]
        sigma_new = sigma - v_new[n,]
        # calculate new gradient
        g = grad(Mu, X, H, sigma_new)
        sigma = sigma_new
        Mu = Mu_new
        v = v_new
        if it % 100 == 0:
            print('\n{0} \t {1} \t {2}'.format(it, np.linalg.norm(g)/g.shape[0], cost(Mu, X, H, sigma)))

    return Mu, sigma

def largest_label_volume(im, bg=-1):
    # find the label of largest regions except background
    vals, counts = np.unique(im, return_counts=True)
    
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    # image[image == -1024] = 0
    binary_image = np.array(image > -320, dtype=np.int8)+1
    binary_image = morphology.closing(binary_image)
    labels = measure.label(binary_image)

    # fix error lung touching outside
    for i in range(labels.shape[0]):
        if labels[i,1,1] == labels[i, int(labels.shape[1]/2), int(labels.shape[2]/3)]:
            image[image == image[0,1,1]] = 0
            print(image[:,1,1])
            binary_image = np.array(image > -320, dtype=np.int8) + 1
            binary_image = morphology.closing(binary_image)
            labels = measure.label(binary_image)
            break
    
    # Pick the pixel in the very corner to determine which label is air.
    background_label = labels[0,1,1]
    binary_image[labels == background_label] = 2

    background_label = labels[0,labels.shape[1]-2,1]
    binary_image[labels == background_label] = 2

    background_label = labels[0,1,labels.shape[2]-2]
    binary_image[labels == background_label] = 2

    background_label = labels[0,labels.shape[1]-2,labels.shape[2]-2]
    binary_image[labels == background_label] = 2
    
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 # Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0
 
    return binary_image

#-------------------------------------------------------------------------------------------#
#                                   normalize data to [0,1]                                 #
# Our values now are ranging from -1000 to 2000. According to the list of HU values, values #
# larger than 300 are not interesting because they are bones. We set threshold between -1000# 
# to 300, and normalize values to [0,1].                                                    #
#-------------------------------------------------------------------------------------------#

MIN_BOUND = -1000.0
MAX_BOUND = 300.0
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

#-------------------------------------------------------------------------------------------#
#                                   normalize data to zero mean                             #
# We can find the mean value from the whole dataset, but it requires lots of computations   #
# As suggested from LUNA16 competition, PIXEL_MEAN = 0.25                                   #
#-------------------------------------------------------------------------------------------#
PIXEL_MEAN = 0.25
def zero_center(image):
    image = image - PIXEL_MEAN
    return image