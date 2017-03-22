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

nslices = 198
data = tf.placeholder(tf.float32, (None,nslices,300,300))

with tf.Session() as sess:


    patient = np.empty((0, nslices, 300, 300), dtype=np.float32)
    temp = np.empty((1, nslices, 300, 300), dtype=np.float32)
    
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
    coord = tf.constant([[0.5,10.4,0.2],[0,-30,0],[100,100,100],[0,0,0]], tf.float32) # Coordinate of points
    stride = tf.constant([1,2,2,2], tf.int16) # Strides
    a = select_module.pixel_selector(data,coord,stride)
    b = sess.run(a, feed_dict={data: patient})

    # show results
    print('RESULTS')
    for i in range(0,patient.shape[0]):
        print('Patient {0}'.format(i))
        # show only the first two slices of each patient
        for j in [0,10]:
            plt.imshow(b[i,j,:,:,0], cmap='gray')
            plt.show()
