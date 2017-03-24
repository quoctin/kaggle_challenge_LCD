from __future__ import print_function
import os
import time
import h5py
import tensorflow as tf
import tensorflow.python as ops
import numpy as np
import csv
import matplotlib.pyplot as plt

# fix random seed in tf
tf.set_random_seed(128)


#-------------------------------------------------------------------------------#
#                                some constants                                 #   
#-------------------------------------------------------------------------------#
PX_SEL_PATH = os.path.abspath('''../exercise/pixel_selector.so''')
INPUT_DATA = os.path.abspath('''../../data/preprocessed_data''')
LABEL_FILE = os.path.abspath('''../../data/stage1_labels.csv''')
CHECKPOINT_PATH = os.path.abspath('checkpoints')
LEARNING_RATE = 0.001
BATCH_SIZE = 5
SKIP_STEP = 5
DROPOUT = 0.75
N_EPOCHS = 20
SLICES = 141
NUM_POINTS = 5
N_PIPELINE = 1
N_TRAIN_SAMPLES = 93

select_module = tf.load_op_library(PX_SEL_PATH)

#-------------------------------------------------------------------------------#
#                                register gradient                              #   
#-------------------------------------------------------------------------------#
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
    coord_grad = ops.zeros_like((NUM_POINTS,3), tf.float32)
    back_grad = ops.reshape(grad,[-1])
    coord_grad_tmp = np.zeros((NUM_POINTS,3), np.float32)
    for i in range(0, NUM_POINTS):
        for j in range(0, 3):
            coord_tmp = np.zeros((NUM_POINTS,3), np.float32)
            coord_tmp[i,j] = 1.0
            coord_tmp = coord + coord_tmp
            tmp_1 = ops.reshape(select_module.pixel_selector(input,coord_tmp,strides),[-1])
            coord_tmp = np.zeros((NUM_POINTS,3), np.float32)
            coord_tmp[i,j] = -1.0
            coord_tmp = coord + coord_tmp
            tmp_2 = ops.reshape(select_module.pixel_selector(input,coord_tmp,strides),[-1])
            tmp = ops.subtract(tmp_1,tmp_2)
            tmp = ops.divide(tmp,2)
            tmp = ops.multiply(tmp,back_grad)
            tmp_3 = np.zeros((NUM_POINTS,3), np.float32)
            tmp_3[i,j] = 1.0
            coord_grad_tmp = coord_grad_tmp + tmp_3*ops.reduce_sum(tmp)

    coord_grad = coord_grad_tmp
    
    return [None,coord_grad,None]

#-------------------------------------------------------------------------------#
#                             network definition                                #  
# Input:                                                                        #
#   nPipeline: number of pixel selectors, default is 1                          #
#   data_shape: the shape of data, (batch,depth,height, width)                  #
#   nPoints: number of sensors for each selector                                #
#-------------------------------------------------------------------------------#
def network_definition(nPipeline=N_PIPELINE,data_shape=(None,None,300,300), nPoints=NUM_POINTS):
    with tf.name_scope('data'):
        X = tf.placeholder(tf.float32, shape=data_shape, name='X_placeholder')
        Y = tf.placeholder(tf.float32, shape=(None,1), name='Y_placeholder')

    wg_out_list = []
    # for each pipeline
    for pl in range(nPipeline):
        pl_str = str(pl)
        with tf.variable_scope('pixel_sel_'+pl_str) as scope:
            coord = tf.get_variable('coordinates', initializer=tf.truncated_normal([nPoints,3], stddev=2.0))
            stride=tf.constant([1,2,2,2], tf.int16) 
            sel_out = select_module.pixel_selector(X, coord, stride)
            # sel_out: [BATCH_SIZE, depth/2, width/2, height/2, nPoints] 
            # ||
            # ||
            # \/
        with tf.variable_scope('conv_'+pl_str) as scope:
            initial_kernel = tf.truncated_normal([1,1,1,nPoints,1], stddev=1)
            kernel = tf.get_variable('kernel',initializer=initial_kernel)
            initial_bias = tf.random_normal([1])
            bias = tf.get_variable('bias',initializer=initial_bias)
            conv = tf.nn.conv3d(sel_out,kernel,[1,1,1,1,1],"SAME")
            conv1 = conv + bias
            # conv1: [BATCH_SIZE, depth/2, width/2, height/2, 1]
            # ||
            # ||
            # \/
        with tf.variable_scope('gaussian_'+pl_str) as scope:
            initial_sigma = 5.0
            sigma = tf.get_variable('sigma', initializer=initial_sigma)
            gaussian = tf.exp(-conv1**2 / (2*sigma*sigma))
            # gaussian_out: [BATCH_SIZE, depth/2, width/2, height/2, 1]
            # ||
            # \/
            initial_weight = tf.truncated_normal([1], stddev=0.1)
            weight = tf.get_variable('weight', initializer=initial_weight)
            weighted_gaussian = tf.multiply(gaussian, weight)
            wg_out_list.append(weighted_gaussian)
    # weighted_gaussian_out: [BATCH_SIZE, depth/2, width/2, height/2, 1]
    # ||
    # ||
    # \/
    with tf.name_scope('sum') as scope:  
        sum_out = tf.zeros_like(wg_out_list[0])
        for i in range(nPipeline):
            sum_out = tf.add(sum_out, wg_out_list[i])         
    # max_pool: [BATCH_SIZE, depth/2, width/2, height/2, 1]
    # ||
    # ||
    # \/
    with tf.name_scope('max') as scope:
        max_pool = tf.reduce_max(sum_out, [1,2,3])

    return [X, Y, conv1, weighted_gaussian, sum_out, max_pool]

#-------------------------------------------------------------------------------#
#                         load training data in batch                           #  
# Input:                                                                        #
#   batch_size: number of patients in a batch                                   #
#   batch_offset: the beginning index of the batch w.r.t the entire data        #
#   slices: number of slices / patient                                          #
#-------------------------------------------------------------------------------#
def next_training_batch(batch_size=5, batch_offset=0, slices=141):
    data = np.zeros((batch_size, slices, 300, 300), dtype=np.float32)
    temp = np.zeros((1, slices, 300, 300), dtype=np.float32)
    labels = np.zeros((batch_size,1), dtype=np.float32)

    # load labels to a dictionary
    # 362 tumors out of 1397 cases
    training_labels = {}
    with open(LABEL_FILE) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            training_labels[row['id']] = float(row['cancer'])
    
    # load training data
    patients = os.listdir(INPUT_DATA)
    patients.sort()
    training_patients = []
    if '.DS_Store' in patients:
        patients.remove('.DS_Store')
    if '._.DS_Store' in patients:
        patients.remove('._.DS_Store')
    for p in patients:
        # take data in the training set
        if p.replace('.hdf5','') in training_labels.keys():
            training_patients.append(p)

    data_size = len(training_patients)
    if  data_size < batch_size:
        raise Exception('Batch_size is larger than available data')
    
    for i in range(batch_size):
        ind = (i + batch_offset) % data_size
        f = h5py.File(INPUT_DATA + "/" + training_patients[ind], "r")
        id = f.attrs['id']
        labels[i,0] = float(training_labels[id])
        data[i, :, :, :] = f['data'][()]
        f.close()
    
    # update and return new batch offset
    batch_offset_new = (batch_offset + batch_size) % data_size

    return [data, labels, batch_offset_new]


#-------------------------------------------------------------------------------#
#                       test the model using training data                      #  
# Input:                                                                        #
#   sess: the current session containing all variables                          #
#   logits: the max pooling layer (only output of this layer is necessary)      #
#   test_epochs: number of testing epochs                                       #
#   test_batch_size: number of patients in a batch                              # 
#-------------------------------------------------------------------------------#
def test_model(sess, X, logits, test_epochs=1, test_batch_size=5):

    batch_offset = 0
    total_correct_pred = 0.0
    n_batches = int(N_TRAIN_SAMPLES / test_batch_size)
    for index in range(test_epochs*n_batches):
        [data, labels, batch_offset] = next_training_batch(batch_size=test_batch_size,\
                                            batch_offset=batch_offset, slices=SLICES)
        logits_batch = sess.run(logits, feed_dict={X: data})
        b_preds = (logits_batch > 0.0)
        b_labels = (labels != 0.0)
        total_correct_pred += np.sum(1.0*(b_labels == b_preds))

    return total_correct_pred/(test_epochs*test_batch_size*n_batches)

#-------------------------------------------------------------------------------#
#                       visualize output of the network                         #  
# Input:                                                                        #
#   sess: the current session containing all variables                          #
#   X: data place holder                                                        #
#   conv1: conv1 op                                                             #
#   weighted_gaussian: weighted gaussian op                                     #
#-------------------------------------------------------------------------------#
def visualize_output(sess, X, conv1, weighted_gaussian, index):
    [data, labels, _] = next_training_batch(batch_size=3,\
                                                batch_offset=0, slices=SLICES)
    
    [out_conv1, out_wg] = sess.run([conv1, weighted_gaussian], feed_dict={X: data})

    with tf.variable_scope('conv_0', reuse=True) as scope:
        bias = tf.get_variable('bias',shape=[1])
    with tf.variable_scope('gaussian_0', reuse=True) as scope:
        weight = tf.get_variable('weight',shape=[1])

    for i in range(0,3):
        for j in range(50,53):
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ax.imshow(data[i,j*2,:,:], cmap='gray') # as the data is subsampled ==> j*2
            #ax.set_title('Original slice')
            ax.set_xlabel('Original')
            ax = fig.add_subplot(132)
            ax.imshow(out_conv1[i,j,:,:,0], cmap='gray')
            ax.set_title('Bias = ' + str(bias.eval()))
            ax.set_xlabel('Conv')
            ax = fig.add_subplot(133)
            ax.imshow(out_wg[i,j,:,:,0], cmap='gray')
            ax.set_title('Weight = ' + str(weight.eval()))
            ax.set_xlabel('Weighted Gaussian')
            plt.tight_layout()

            fname = 'iter_' + str(index+1) + '_patient_' + str(i+1) + '_slice_' + str(j+1) + '.png'
            plt.savefig('fig/'+fname)


def main():
    # reset the graph before defining the network
    tf.reset_default_graph()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    global_batch_offset = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_batch_offset')

    [X, Y, conv1, weighted_gaussian, sum_out, max_pool] = network_definition()
    

    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=max_pool)
    loss = tf.reduce_mean(entropy)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,\
            global_step=global_step)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHECKPOINT_PATH+'/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        initial_step = global_step.eval()
        batch_offset = global_batch_offset.eval()

        n_batches = int(N_TRAIN_SAMPLES / BATCH_SIZE)

        for index in range(initial_step, N_EPOCHS*n_batches):
            
            [data, labels, batch_offset] = next_training_batch(batch_size=BATCH_SIZE,\
                                            batch_offset=batch_offset, slices=SLICES)
            start_time = time.time()
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: data, Y: labels})
            print('Iteration {0} with Loss {1}'.format(index+1, loss_batch))
            #print("Epoch time: {0} seconds".format(time.time() - start_time))

            # test the model
            accuracy = test_model(sess, X, max_pool, test_batch_size=10)
            print('Iteration {0} with accuracy {1}'.format(index+1, accuracy))

            # visualize output after SKIP_STEP
            if (index+1) % SKIP_STEP == 0:
                visualize_output(sess, X, conv1, weighted_gaussian, index)

            # update the global_batch_offset  
            sess.run(global_batch_offset.assign(batch_offset))

            # save model after SKIP_STEP
            if (index+1) % SKIP_STEP == 0:
                saver.save(sess, CHECKPOINT_PATH+'/checkpoint', index)

if __name__ == '__main__':
    main()