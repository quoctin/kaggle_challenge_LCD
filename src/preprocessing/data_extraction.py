import os
from preprocessing import *
import h5py

# Some constants
SERVER = 1

if SERVER == 1:
    INPUT_FOLDER = '/home/quoctin/qnapshare/Datasets/DSB2017/stage1'
    OUTPUT_FOLDER = '/home/quoctin/qnapshare/Datasets/DSB2017/extracted_data'
else:
    INPUT_FOLDER='../data/exceptions'
    OUTPUT_FOLDER='../data/extracted_data'

patients = os.listdir(INPUT_FOLDER)
patients.sort()
if '.DS_Store' in patients:
    patients.remove('.DS_Store')
if '._.DS_Store' in patients:
    patients.remove('._.DS_Store')
    
print('There are {0} patients in the folder.'.format(len(patients)))

for i,p in enumerate(patients):
    print('\nProcessing patient: {0}\n'.format(p))
    patient = load_scan(INPUT_FOLDER + '/' + p)
    patient_pixels = convert_pix2hu(patient)
    pix_resampled = resample(patient_pixels, patient, new_spacing=[1.0, 1.0, 1.0])
    # save resampled data
    f = h5py.File(OUTPUT_FOLDER + "/" + p + ".hdf5", "w")
    f.attrs['id'] = p
    f.create_dataset('data', data=pix_resampled)
    f.close()
