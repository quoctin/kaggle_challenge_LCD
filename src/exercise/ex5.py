import os
import h5py
import csv
import tensorflow as tf
import tensorflow.python as ops
import matplotlib.pyplot as plt
import numpy as np

select_module = tf.load_op_library('/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/src/exercise/pixel_selector.so')

# declare some variables
INPUT_FOLDER = '/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/data/normalized_data'
LABEL_FILE = '/Users/emanuele/Google Drive/Dottorato/Software/kaggle_challenge_LCD/data/stage1_labels.csv'

SLICES = 198
NUM_POINTS = 5
WIDTH = 300
HEIGHT = 300
STRIDE = [1,2,2,2]


@ops.RegisterGradient("PixelSelector")
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
    shape = ops.shape(coord)
    coord_grad = ops.zeros_like(shape)
    back_grad = ops.reshape(grad,[-1])
    
    for i in range(0, NUM_POINTS):
        for j in range(0, 3):
            coord_tmp = np.zeros((NUM_POINTS,3))
            coord_tmp[i,j] = 1.0
            coord_tmp = coord + coord_tmp
            tmp_1 = ops.reshape(select_module.pixel_selector(input,coord_tmp,strides),[-1])
            coord_tmp = np.zeros((NUM_POINTS,3))
            coord_tmp[i,j] = -1.0
            coord_tmp = coord + coord_tmp
            tmp_2 = ops.reshape(select_module.pixel_selector(input,coord_tmp,strides),[-1])
            tmp = ops.subtract(tmp_1,tmp_2)
            tmp = ops.divide(tmp,2)
            tmp = ops.multiply(tmp,back_grad)
            tmp_3 = np.zeros((NUM_POINTS,3)) # TO CHECK
            tmp_3[i,j] = ops.reduce_sum(tmp) # TO CHECK
            coord_grad = coord_grad + tmp_3 # TO CHECK

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
    patient = np.append(patient, temp, axis=0)
    f.close()
    str = patients[i]
    str = str.replace(".hdf5","")
    label = np.append(label, [[labels[str]]], axis=0)
# +++++++++++++++++++++++++++++++++++++++++
# 'patient' and 'label' are used as dataset
# +++++++++++++++++++++++++++++++++++++++++


# COMPUTATIONAL GRAPH
initial = tf.truncated_normal([NUM_POINTS,3], stddev=10.0)
coord = tf.Variable(initial) # Coordinate of points
stride = tf.constant(STRIDE, tf.int16) # Strides
data = tf.placeholder(tf.float32, (None,SLICES, WIDTH, HEIGHT))
y = tf.placeholder(tf.float32, [None, 1])
initial_kernel = tf.truncated_normal([1,1,1,NUM_POINTS,1], stddev=0.1)
kernel = tf.Variable(initial_kernel)
initial_bias = tf.random_normal([1])
bias = tf.Variable(initial_bias)
init_op = tf.global_variables_initializer()

# NETWORK
x = select_module.pixel_selector(data,coord,stride)
conv = tf.nn.conv3d(x,kernel,[1,1,1,1,1],"SAME")
conv1 = conv + bias
out = tf.reduce_sum(conv1,[1,2,3])
print(y.shape)
print(out.shape)
entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=out)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
    sess.run(init_op)
    _, loss_batch = sess.run([optimizer, loss], feed_dict={data: patient, y: label})
    
    # show results
    print('VISUALIZING RESULTS')
    for i in range(0,patient.shape[0]):
        print('Patient {0}'.format(i))
        # show only the first two slices of each patient
        for j in [0,10]:
            plt.imshow(conv1[i,j,:,:,0], cmap='gray')
            plt.show()
