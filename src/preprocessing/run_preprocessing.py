import os
from preprocessing import *
import matplotlib.pyplot as plt
plt.close("all")

# Some constants
SERVER = 1
if SERVER == 1:
    INPUT_FOLDER = '/home/quoctin/qnapshare/Datasets/DSB2017/extracted_data'
else:
    INPUT_FOLDER = '../data/extracted_data'
OUTPUT_FOLDER = 'auxilaries'

patients = os.listdir(INPUT_FOLDER)
patients.sort()
if '.DS_Store' in patients:
    patients.remove('.DS_Store')
if '._.DS_Store' in patients:
    patients.remove('._.DS_Store')

print('There are {0} patients in the folder.'.format(len(patients)))

#------------------------------------------------------------------------#
#                           Find best slice cut                          #
#------------------------------------------------------------------------#

best_slice_cut(INPUT_FOLDER, OUTPUT_FOLDER, loadHX = 0)

#------------------------------------------------------------------------#
#                  Lung segmentation and normalize data                  #
#------------------------------------------------------------------------#

# change the folder to store larger data

if SERVER == 1:
    OUTPUT_FOLDER = '/home/quoctin/qnapshare/Datasets/DSB2017/preprocessed_data'
else:
    OUTPUT_FOLDER = '../data/preprocessed_data'

finds = pickle.load(open('auxilaries/finds.p', 'rb'))
normalized_shape = [finds[0,2], 300, 300]

new_sizes = np.zeros((len(patients), 3))
for i,p in enumerate(patients):
    # save resampled data
    f = h5py.File(INPUT_FOLDER + "/" + p, "r")
    data = f['data']
    id = f.attrs['id']
    f.close

    # cut the slides not containing lungs
    data = data[finds[i,0]:finds[i,1], :, :]

    # lung segmentation
    segmented_data = segment_lung_mask(data)
    segmented_data = segmented_data*data

    # rescaling to make them the same size
    resize_factor = np.array(normalized_shape) / segmented_data.shape
    resize_factor[0] = 1.0
    ndata = scipy.ndimage.interpolation.zoom(segmented_data, resize_factor, mode='nearest')

    # normalize to [0,1] and 0.25 mean
    ndata = zero_center(normalize(ndata))

    print('\nProcessing {0}: dest. size = {1}\n'.format(p, ndata.shape))

    f = h5py.File(OUTPUT_FOLDER + "/" + p, "w")
    f.attrs['id'] = id
    f.create_dataset('data', data=ndata)
    f.close()

print('\nDone\n')

