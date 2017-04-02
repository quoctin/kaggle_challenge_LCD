import os
import h5py
import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle

# declare some variables
SERVER = 1
if SERVER == 1:
    INPUT_FOLDER = '/home/quoctin/qnapshare/Datasets/DSB2017/preprocessed_data'
else:
    INPUT_FOLDER = '../data/preprocessed_data'

LABEL_FILE = '../data/stage1_labels.csv'
TRAINING_FOLDER = '/home/quoctin/qnapshare/Datasets/DSB2017/training_set'
VALIDATION_FOLDER = '/home/quoctin/qnapshare/Datasets/DSB2017/validation_set'

# load labels to a dictionary
labels = {}
with open(LABEL_FILE) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        labels[row['id']] = float(row['cancer'])

# load all data
patients = os.listdir(INPUT_FOLDER)
patients.sort()
if '.DS_Store' in patients:
    patients.remove('.DS_Store')
if '._.DS_Store' in patients:
    patients.remove('._.DS_Store')

# loop over patients
for i,p in enumerate(patients):
    f = h5py.File(INPUT_FOLDER + "/" + p, "r")
    id = f.attrs['id']
    data = f['data'][()]
    print('\nProcessing ', id)
    if id in labels.keys():
        ff = h5py.File(TRAINING_FOLDER + "/" + p, "w")
        ff.attrs['id'] = id
        ff.create_dataset('data', data=data)
        ff.close()
    else:
        ff = h5py.File(VALIDATION_FOLDER + "/" + p, "w")
        ff.attrs['id'] = id
        ff.create_dataset('data', data=data)
        ff.close()
