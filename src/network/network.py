from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np

#------------------------------------------------------------#
#                       some constants                       #   
#------------------------------------------------------------#
PX_SEL_PATH = '../exercise/pixel_selector.so'
INPUT_DATA = '../../data/preprocessed_data'
LEARNING_RATE = 0.001
BATCH_SIZE = 1
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 1

module = tf.load_op_library(PX_SEL_PATH)

#---------------------------------------------------------------#
#                       some constants                          #  
# Input:                                                        #
#   nPipeline: number of pixel selectors, default is 1          #
#   data_shape: the shape of data, (batch,depth,height, width)  #
#---------------------------------------------------------------#
def network_definition(nPipeline=1,data_shape=(None,None,300,300), nPoints=5):
    with tf.name_scope('data'):
        X = tf.placeholder(tf.float32, shape=data_shape, name='X_placeholder')
        Y = tf.placeholder(tf.float32, shape=(None,1), name='Y_placeholder')

    # for each pipeline
    #for pl in range(nPipeline):
    #    pl_str = str(pl)
    with tf.variable_scope('pipeline'):
        coord = tf.get_variable('coordinates', shape=[nPoints,3], 
        initializer=tf.truncated_normal([nPoints,3], stddev=5.0))
        stride=tf.constant([1,2,2,2], tf.float32) 
        s = module.pixel_selector(X, coord, stride)
        return s
    

def main():
    # reset the graph before defining the network
    tf.reset_default_graph()
    
    SLICES=198
    patient = np.empty((0, SLICES, 300, 300))
    temp = np.empty((1, SLICES, 300, 300))
    label = np.empty((0,1))

    print('LOADING TRAINING DATA')
    for i,p in enumerate(patients):
        f = h5py.File(INPUT_DATA + "/" + patients[i], "r")
        temp[0,:,:,:] = f['data'][()]
        patient = np.append(patient, temp, axis=0)
        f.close()
        str = patients[i]
        str = str.replace(".hdf5","")
        label = np.append(label, [[labels[str]]], axis=0)

    s = network_definition()

    with tf.Session() as sess:
        out = sess.run(s, feed_dict={X: patient})
        print('VISUALIZING RESULTS')
        for i in range(0,patient.shape[0]):
            print('Patient {0}'.format(i))
            # show only the first two slices of each patient
            for j in [0,10]:
                plt.imshow(out[i,j,:,:,0], cmap='gray')
                plt.show()


if __name__ == '__main__':
    main()