import os
import h5py
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

replicate_module = tf.load_op_library('/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/src/exercise/replicate_data.so')

# declare some variables
INPUT_FOLDER = '/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/data/normalized_data'
LABEL_FILE = '/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/data/stage1_labels.csv'


num_replicas = 6

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

data = tf.placeholder(tf.int16, [None,114,300,300])

with tf.Session() as sess:

    patient = np.empty((0, 114, 300, 300))
    temp = np.empty((1, 114, 300, 300))

    # create dataset of patients and store as a 4D numpy array
    # [batch_size, depth, width, height]
    print('TRAINING DATA')
    for i,p in enumerate(patients):
        print('Patient {0}'.format(i))
        f = h5py.File(INPUT_FOLDER + "/" + patients[i], "r")
        temp[0,:,:,:] = f['data'][()]
        patient = np.append(patient, temp, axis=0)
        f.close()

    # using library
    b = tf.constant(num_replicas) # Number of replicas
    a = replicate_module.replicate_data(patient,b)
    sess.run(a, feed_dict={data: patient})
    print('Shape of the output tensor {0}'.format(a.shape))

    # show results
    for i in range(0,patient.shape[0]-1):
        print('Patient {0}'.format(i))
        # show only the first two slices of each patient
        for j in range(0,2):
            plt.imshow(a[i,j,:,:,num_replicas-1].eval(), cmap='gray')
            plt.show()
