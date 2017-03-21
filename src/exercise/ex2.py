import os
import h5py
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

replicate_module = tf.load_op_library('/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/src/exercise/replicate_mat.so')

# declare some variables
INPUT_FOLDER = '/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/data/normalized_data'
LABEL_FILE = '/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/data/stage1_labels.csv'

# load labels to a dictionary
labels = {}
with open(LABEL_FILE) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        labels[row['id']] = float(row['cancer'])

# load training data
patients = os.listdir(INPUT_FOLDER)
patients.sort()
if '.DS_Store' in patients:
    patients.remove('.DS_Store')
if '._.DS_Store' in patients:
    patients.remove('._.DS_Store')

with tf.Session() as sess:
    
    for i,p in enumerate(patients):
        # Take first patient
        print('Patient {0}'.format(i))
        f = h5py.File(INPUT_FOLDER + "/" + patients[i], "r")
        patient = f['data'][()] # Converting H5PY file to numpy array
        f.close()
        b = tf.constant(6) # Number of replicas
        a = replicate_module.replicate_mat(patient,b)
        sess.run(a)
        print(a.shape)
        # Show for each patient the first 10 slices (using the the 6-th replica)
        for j in range(0,10):
            plt.imshow(a[j,:,:,5].eval(), cmap='gray')
            plt.show()
