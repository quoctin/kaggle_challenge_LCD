import os
import h5py
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

select_module = tf.load_op_library('/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/src/exercise/pixel_selector.so')

# declare some variables
INPUT_FOLDER = '/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/data/normalized_data'
LABEL_FILE = '/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/data/stage1_labels.csv'

SLICES = 198

# load labels to a dictionary
# 362 tumors out of 1397 cases
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


patient = np.empty((0, SLICES, 300, 300))
temp = np.empty((1, SLICES, 300, 300))
label = np.empty((0,1))

# create dataset of patients and store as a 4D numpy array
# [batch_size, depth, width, height]
print('LOADING TRAINING DATA')
for i,p in enumerate(patients):
    f = h5py.File(INPUT_FOLDER + "/" + patients[i], "r")
    temp[0,:,:,:] = f['data'][()]
    patient = np.append(patient, temp, axis=0)
    f.close()
    str = patients[i]
    str = str.replace(".hdf5","")
    label = np.append(label, [[labels[str]]], axis=0)
# +++++++++++++++++++++++++++++++++++++++++
# 'patient' and 'label' are used as dataset
# +++++++++++++++++++++++++++++++++++++++++


# COMPUTATIONAL GRAPH
initial = tf.truncated_normal([5,3], stddev=10.0)
coord = tf.Variable(initial) # Coordinate of points
stride = tf.constant([1,2,2,2], tf.int16) # Strides
data = tf.placeholder(tf.float32, (None,SLICES,300,300))

with tf.Session() as sess:
    sess.run(coord.initializer)
    x = select_module.pixel_selector(data,coord,stride)

    # TODO: add classification layer
    # x_reshape = tf.reshape(x_reshape,)
    
    out = sess.run(x, feed_dict={data: patient})

    # show results
    print('VISUALIZING RESULTS')
    for i in range(0,patient.shape[0]):
        print('Patient {0}'.format(i))
        # show only the first two slices of each patient
        for j in [0,10]:
            plt.imshow(out[i,j,:,:,0], cmap='gray')
            plt.show()
