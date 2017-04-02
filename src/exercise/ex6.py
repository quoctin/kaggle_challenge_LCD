# Modifications for Tensorflow 1.0.1

import os
import h5py
import csv
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

select_module = tf.load_op_library('./pixel_selector.so')

# declare some variabless
INPUT_FOLDER = '../../../Datasets/small_kaggle'
LABEL_FILE = '../../data/stage1_labels.csv'

SLICES = 141
NUM_POINTS = 5
WIDTH = 300
HEIGHT = 300
STRIDE = [1,2,2,2]

LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 100

@tf.RegisterGradient("PixelSelector")
def _pixel_selector_grad(op, grad):
    """The gradients for 'pixel_selector'.
        
        Args:
        op: The 'pixel_selector' operation we want to differentiate.
        grad: Gradient with respect to the output of the 'pixel_selector' op.
        
        Returns:
        Gradients with respect to the coordinates of points of interest for 'pixel_selector'.
        """
    input = op.inputs[0]
    coord = op.inputs[1]
    strides = op.inputs[2]
    coord_grad = tf.zeros_like((NUM_POINTS,3), tf.float32)
    back_grad = tf.reshape(grad,[-1])
    coord_grad_tmp = np.zeros((NUM_POINTS,3), np.float32)
    for i in range(0, NUM_POINTS):
        for j in range(0, 3):
            coord_tmp = np.zeros((NUM_POINTS,3), np.float32)
            coord_tmp[i,j] = 1.0
            coord_tmp = coord + coord_tmp
            tmp_1 = tf.reshape(select_module.pixel_selector(input,coord_tmp,strides),[-1])
            coord_tmp = np.zeros((NUM_POINTS,3), np.float32)
            coord_tmp[i,j] = -1.0
            coord_tmp = coord + coord_tmp
            tmp_2 = tf.reshape(select_module.pixel_selector(input,coord_tmp,strides),[-1])
            tmp = tf.subtract(tmp_1,tmp_2)
            tmp = tf.divide(tmp,2)
            tmp = tf.multiply(tmp,back_grad)
            tmp_3 = np.zeros((NUM_POINTS,3), np.float32)
            tmp_3[i,j] = 1.0
            coord_grad_tmp = coord_grad_tmp + tmp_3*tf.reduce_sum(tmp)

    coord_grad = coord_grad_tmp
    
    return [None,coord_grad,None]

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


patient = np.empty((0, SLICES, WIDTH, HEIGHT))
temp = np.empty((1, SLICES, WIDTH, HEIGHT))
label = np.empty((0,1))

# create dataset of patients and store as a 4D numpy array
# [batch_size, depth, width, height]
print('LOADING TRAINING DATA')
for i,p in enumerate(patients):
    f = h5py.File(INPUT_FOLDER + "/" + patients[i], "r")
    temp[0,:,:,:] = f['data'][()]
    f.close()
    str = patients[i]
    str = str.replace(".hdf5","")
    if not (str in labels.keys()):
	continue
    patient = np.append(patient, temp, axis=0)
    label = np.append(label, [[labels[str]]], axis=0)
# +++++++++++++++++++++++++++++++++++++++++
# 'patient' and 'label' are used as dataset
# +++++++++++++++++++++++++++++++++++++++++


# COMPUTATIONAL GRAPH
initial = tf.truncated_normal([NUM_POINTS,3], stddev=2.0)
coord = tf.Variable(initial) # Coordinate of points
stride = tf.constant(STRIDE, tf.int16) # Strides
data = tf.placeholder(tf.float32, (None,SLICES, WIDTH, HEIGHT))
y = tf.placeholder(tf.float32, [None, 1])
initial_kernel = tf.truncated_normal([1,1,1,NUM_POINTS,1], stddev=0.1)
kernel = tf.Variable(initial_kernel)
initial_bias = tf.random_normal([1])
bias = tf.Variable(initial_bias)

# NETWORK
x = select_module.pixel_selector(data,coord,stride)
conv = tf.nn.conv3d(x,kernel,[1,1,1,1,1],"SAME")
conv1 = conv + bias
out = tf.reduce_sum(conv1,[1,2,3])
entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=out)
loss = tf.reduce_mean(entropy)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,\
            global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('TRAINING')
    for index in range(0, EPOCHS-1):
        start_time = time.time()
        _, loss_batch = sess.run([optimizer, loss], feed_dict={data: patient, y: label})
        print('Epoch {0} with Loss {1}'.format(index+1,loss_batch))
        print(coord.eval())
        print("Epoch time: {0} seconds".format(time.time() - start_time))

    print("Optimization Finished!") # should be around 0.35 after 25 epochs

    # show results
    print('VISUALIZING RESULTS')
    result = sess.run(conv1, feed_dict={data: patient})
    for i in range(0,patient.shape[0]):
        print('Patient {0}'.format(i))
        # show only the first two slices of each patient
        for j in [0,10]:
            plt.imshow(result[i,j,:,:,0], cmap='gray')
            plt.show()
