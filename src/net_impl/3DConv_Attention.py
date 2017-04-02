from __future__ import print_function
import os
import time
import h5py
import tensorflow as tf
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pkg_resources as pkr
import pickle

#-------------------------------------------------------------------------------#
#                                some constants                                 #
#-------------------------------------------------------------------------------#
INPUT_DATA = os.path.abspath('''../../data/preprocessed_data''')
LABEL_FILE = os.path.abspath('''/home/quoctin/qnapshare/Datasets/DSB2017/stage1_labels.csv''')
CHECKPOINT_PATH = os.path.abspath('checkpoints')
LOSS_FILE = 'logging_loss.dat'
ACC_FILE = 'logging_acc.dat'
LEARNING_RATE = 0.001
BATCH_SIZE = 1
SKIP_STEP = 5
DROPOUT = 0.75
N_EPOCHS = 51
SLICES = 141
NUM_POINTS = 5
N_PIPELINE = 1
N_TRAIN_SAMPLES = 100

#-------------------------------------------------------------------------------#
#                             network definition                                #  
# Input:                                                                        #                       #
#   data_shape: the shape of data, (batch,depth,height, width)                  #                             #
#-------------------------------------------------------------------------------#
def network_definition(data_shape=(None,SLICES,300,300)):
    params = {}
    with tf.name_scope('data'):
        X = tf.placeholder(tf.float32, shape=data_shape, name='X_placeholder')
        Y = tf.placeholder(tf.float32, shape=(None,1), name='Y_placeholder')

    with tf.variable_scope('conv0') as scope:
        X_expand_dim = tf.expand_dims(X,4)
        out_filters = 1
        initial_kernel = tf.truncated_normal([4,4,4,1,out_filters], stddev=0.1)
        kernel = tf.get_variable('kernel',initializer=initial_kernel)
        kernel_abs = tf.abs(kernel)
        normalized_kernel = tf.divide(kernel_abs, tf.reduce_sum(kernel_abs))
        initial_biases = tf.random_normal([out_filters], stddev=0.1)
        biases = tf.get_variable('biases',initializer=initial_biases)
        conv0_0 = tf.nn.conv3d(X_expand_dim, normalized_kernel,[1,2,2,2,1],"SAME")
        conv0_1 = tf.nn.bias_add(conv0_0, biases)
        pool0 = tf.nn.max_pool3d(conv0_1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
        # pool0: [BATCH_SIZE, depth/4, width/4, height/4, out_filters] 
        # ||
        # ||
        # \/
    with tf.variable_scope('gaussian') as scope:
        initial_sigma = 0.01
        sigma = tf.get_variable('sigma', initializer=initial_sigma)
        squared_neg = tf.negative(tf.square(pool0))
        sigma_squared = tf.multiply(tf.square(sigma),2)
        gaussian = tf.exp(squared_neg / sigma_squared)
        # gaussian: [BATCH_SIZE, depth/4, width/4, height/4, out_filters] 
        # ||
        # ||
        # \/
    with tf.variable_scope('conv1') as scope:
        out_filters = 1
        initial_kernel = tf.truncated_normal([2,2,2,1, out_filters], stddev=0.1)
        kernel = tf.get_variable('kernel', initializer=initial_kernel)
        kernel_abs = tf.abs(kernel)
        normalized_kernel = tf.divide(kernel_abs, tf.reduce_sum(kernel_abs))
        initial_biases = tf.random_normal([out_filters], stddev=0.1)
        biases = tf.get_variable('biases', initializer=initial_biases)
        conv1_0 = tf.nn.conv3d(gaussian, normalized_kernel, [1,2,2,2,1], "SAME")
        conv1_1 = tf.nn.bias_add(conv1_0, biases)
        pool1 = tf.nn.max_pool3d(conv1_1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
        # pool1: [BATCH_SIZE, depth/16, width/16, height/16, out_filters] 
        # ||
        # ||
        # \/
    prev_layer = pool1
    prev_shape = prev_layer.get_shape().as_list()
    npixels = np.prod(prev_shape[1:]) # dimension 0 is BATCH_SIZE
    prev_layer_flat = tf.reshape(prev_layer, [-1,npixels]) # size = [BATCH_SIZE, npixels]
    npairs = npixels**2

    indices = tf.constant(np.array(range(npixels)), dtype=tf.float32)
    i1,i2 = tf.meshgrid(indices, indices)
    i1 = tf.reshape(i1, [npairs,1])
    i2 = tf.reshape(i2, [npairs,1])
    nbatches = tf.shape(X)[0]           # a tensor which we don't know the value before running
    i1 = tf.tile(i1, [nbatches, 1])                                     # size = [BATCH_SIZE*npairs, 1]
    i2 = tf.tile(i2, [nbatches, 1])                                     # size = [BATCH_SIZE*npairs, 1]
    batch_indices = tf.expand_dims(tf.range(nbatches), axis=1)          # size = [BATCH_SIZE, 1]
    batch_indices = tf.tile(batch_indices, [1,npairs])                  # size = [BATCH_SIZE, npairs]
    batch_indices = tf.reshape(batch_indices, [-1,1])                   # size = [BATCH_SIZE*npairs, 1]
    batch_indices = tf.cast(batch_indices, dtype=tf.float32)
    vi1 = tf.concat([batch_indices, i1], axis=1)                        # size = [BATCH_SIZE*npairs, 2]
    vi2 = tf.concat([batch_indices, i2], axis=1)                        # size = [BATCH_SIZE*npairs, 2]

    v1 = tf.gather_nd(prev_layer_flat, tf.cast(vi1, dtype=tf.int32))    # size = [BATCH_SIZE*npairs, 1]
    v1 = tf.expand_dims(v1,axis=1)
    v2 = tf.gather_nd(prev_layer_flat, tf.cast(vi2, dtype=tf.int32))    # size = [BATCH_SIZE*npairs, 1]
    v2 = tf.expand_dims(v2,axis=1)
    input_pairs = tf.concat([v1,v2], axis=1)                            # size = [BATCH_SIZE*npairs, 2]
    
    with tf.variable_scope('attention/layer_1'):
        initial_weights = tf.truncated_normal([2, 1], stddev=0.1)
        weights = tf.get_variable('weights', initializer=initial_weights)
        initial_bias = tf.random_normal([1], stddev=0.1)
        bias = tf.get_variable('bias', initializer=initial_bias)
        output_layer1 = tf.nn.bias_add(tf.matmul(input_pairs, weights), bias)
        # output_layer1: [BATCH_SIZE*npairs, 1] 
        # ||
        # ||
        # \/
    with tf.variable_scope('attention/layer_2'):
        initial_weight = tf.truncated_normal([1,1], stddev=0.1)
        weight = tf.get_variable('weight', initializer=initial_weight)
        initial_bias = tf.random_normal([1], stddev=0.1)
        bias = tf.get_variable('bias', initializer=initial_bias)
        output_layer2 = tf.nn.bias_add(tf.matmul(output_layer1, weight), bias)
        # output_layer2: [BATCH_SIZE*npairs, 1] 
        # ||
        # ||
        # \/
    # multiply by distances
    dist = tf.square(i1 - i2)                                         # size = [BATCH_SIZE*npairs, 1]
    input_dists = tf.reshape(dist, [-1, 1])                           # size = [BATCH_SIZE*npairs, 1]

    with tf.name_scope('attention/output'):         
        scores = tf.negative(tf.multiply(input_dists, output_layer2))
        scores_reshape = tf.reshape(scores, [-1, npairs])             # size = [BATCH_SIZE, npairs]
        output_softmax = tf.nn.softmax(scores_reshape)                # size = [BATCH_SIZE, npairs]
        prev_layer_rescale = tf.tile(prev_layer_flat, [1,npixels])    # size = [BATCH_SIZE, npairs]
        mul_scores = tf.multiply(output_softmax, prev_layer_rescale)  # size = [BATCH_SIZE, npairs]
    
        output_attention = tf.reduce_sum(tf.reshape(mul_scores, [-1, npixels, npixels]), axis=2)
        new_shape = [-1, npixels, 1] # None = -1
        output_attention_reshape = tf.reshape(output_attention, new_shape) # size = [BATCH_SIZE, npixels]
        visual_output_attention = tf.reshape(output_attention_reshape, [-1,prev_shape[1],prev_shape[2]*prev_shape[3]]) # size = [BATCH_SIZE, depth/16, width/16, height/16, out_filters] 
        visual_output_attention = tf.reshape(visual_output_attention, [-1,prev_shape[1],prev_shape[2],prev_shape[3]])

    with tf.variable_scope('output'):
        initial_weight = tf.truncated_normal([1], stddev=0.1)
        weight = tf.get_variable('weight', initializer=initial_weight)
        initial_bias = tf.random_normal([1], stddev=0.1)
        bias = tf.get_variable('bias', initializer=initial_bias)
        output_max = tf.reduce_max(output_attention_reshape, axis=1)
        logits = tf.add(tf.multiply(output_max,weight), bias)
        #logits = tf.expand_dims(logits,axis=1)
    
    # save necessary parameters
    params['conv0_1'] = conv0_1
    params['pool0'] = pool0
    params['gaussian'] = gaussian
    params['conv1_1'] = conv1_1
    params['pool1'] = pool1
    params['visual_output_attention'] = visual_output_attention
    params['logits'] = logits
    
    return X,Y,params


#-------------------------------------------------------------------------------#
#                         load training data in batch                           #  
# Input:                                                                        #
#   batch_size: number of patients in a batch                                   #
#   batch_offset: the beginning index of the batch w.r.t the entire data        #
#   slices: number of slices / patient                                          #
#-------------------------------------------------------------------------------#
def next_training_batch(batch_size=5, batch_offset=0, slices=141):
    data = np.zeros((batch_size, slices, 300, 300), dtype=np.float32)
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
def test_model(sess, X, params, test_epochs=1, test_batch_size=5):

    batch_offset = 0
    total_correct_pred = 0.0
    n_batches = int(N_TRAIN_SAMPLES / test_batch_size)
    for index in range(test_epochs*n_batches):
        [data, labels, batch_offset] = next_training_batch(batch_size=test_batch_size,\
                                            batch_offset=batch_offset, slices=SLICES)
        logits_batch = sess.run(params['logits'], feed_dict={X: data})
        #(logits_batch)
        b_preds = (logits_batch > 0.0)
        b_labels = (labels > 0.0)
        total_correct_pred += np.sum(1.0*(b_labels == b_preds))

    return total_correct_pred/(test_epochs*test_batch_size*n_batches)


#-------------------------------------------------------------------------------#
#                       visualize output of the network                         #  
# Input:                                                                        #
#   sess: the current session containing all variables                          #
#   X: data place holder                                                        #
#   params: parameters of the model                                             #
#   index: the iteration index                                                  #
#-------------------------------------------------------------------------------#
def visualize_output(sess, X, params, index):
    [data, labels, _] = next_training_batch(batch_size=3,\
                                                batch_offset=5, slices=SLICES)
    
    [conv0, gaussian, conv1, attention] = sess.run([params['conv0'], \
                                                        params['gaussian'],\
                                                        params['conv1'],\
                                                        params['visual_output_attention']], \
                                                        feed_dict={X: data})
    with tf.variable_scope('conv0', reuse=True):
        bias_conv0 = tf.get_variable('biases')
    with tf.variable_scope('conv1', reuse=True):
        bias_conv1 = tf.get_variable('biases')
    with tf.variable_scope('conv1', reuse=True):
        sigma = tf.get_variable('sigma')

    for i in range(0,3):
        for j in range(5,6):
            fig = plt.figure()
            
            ax = fig.add_subplot(231)
            ax.imshow(data[i,j*16,:,:], cmap='gray') # as the data is subsampled ==> j*16
            ax.set_xlabel('Original')

            ax = fig.add_subplot(232)
            ax.imshow(conv0[i,j*4,:,:,0], cmap='gray')
            ax.set_title('Bias = ' + str(bias_conv0.eval()))
            ax.set_xlabel('After Conv 0')

            ax = fig.add_subplot(233)
            ax.imshow(gaussian[i,j,:,:,0], cmap='gray')
            ax.set_title('Sigma = ' + str(sigma.eval()))
            ax.set_xlabel('After Gaussian')

            ax = fig.add_subplot(234)
            ax.imshow(conv1[i,j,:,:,0], cmap='gray')
            ax.set_title('Bias = ' + str(bias_conv1.eval()))
            ax.set_xlabel('After Conv 1')

            ax = fig.add_subplot(235)
            ax.imshow(attention[i,j,:,:], cmap='gray')
            #ax.set_title('Bias = ' + str(bias_conv1.eval()))
            ax.set_xlabel('After Attention')

            ax = fig.add_subplot(236)
            max_value = tf.reduce_max(attention[i,:,:,:])
            max_indices = tf.where(tf.equal(attention[i,:,:,:], max_value))
            ax.imshow(data[i,max_indices[0]*16,:,:], cmap='gray')
            pos_x = max_indices[1]*16
            pos_y = max_indices[2]*16
            rect = matplotlib.patches.Rectangle((pos_x, pos_y), width=16, height=16, color='red')
            ax.add_patch(rect)
            ax.set_xlabel('Localized possible tumor')

            plt.tight_layout()

            fname = 'iter_' + str(index+1) + '_patient_' + str(i+1) + '_slice_' + str(j+1) + '.png'
            plt.savefig('fig/'+fname)
            plt.close()


def main():
    # reset the graph before defining the network
    tf.reset_default_graph()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    global_batch_offset = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_batch_offset')

    X, Y, params = network_definition()

    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=params['logits'])
    loss = tf.reduce_mean(entropy)
    
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,\
            global_step=global_step)
    
    loss_all = []
    acc_all = []
    if os.path.isfile(LOSS_FILE):
        f = open(LOSS_FILE, 'rb')
        loss_all = pickle.load(f)
        f.close()
    if os.path.isfile(ACC_FILE):
        f = open(ACC_FILE, 'rb')
        acc_all = pickle.load(f)
        f.close()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHECKPOINT_PATH+'/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            print('\n::::::::::Load the model::::::::::\n')
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        initial_step = global_step.eval()
        batch_offset = global_batch_offset.eval()

        n_batches = int(N_TRAIN_SAMPLES / BATCH_SIZE)

        for index in range(initial_step, N_EPOCHS*n_batches):
            
            [data, labels, batch_offset] = next_training_batch(batch_size=BATCH_SIZE,\
                                            batch_offset=batch_offset, slices=SLICES)

            start_time = time.time()
            _, loss_batch, logits = sess.run([optimizer, loss, params['logits']], feed_dict={X: data, Y: labels})
            loss_all.append(loss_batch)
            
            print('\n::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n')
            print('Interation {0}\n'.format(index+1))            
            print('\tAverage Loss: {0}\n'.format(np.average(np.asarray(loss_all))))
            print('\tLogits: \n{0}\n'.format(logits))
            print('\tLabels: \n{0}\n'.format(labels))
            print("\tTime: {0} sec\n".format(time.time() - start_time))
            
            # test the model
            if (index+1) % SKIP_STEP == 0:
                accuracy = test_model(sess, X, params, test_batch_size=BATCH_SIZE)
                print('====>Accuracy {0}\n'.format(accuracy))
                acc_all.append(accuracy)
                avg_accuracy = np.average(np.asarray(acc_all))
                print('====>Average accuracy {0}\n'.format(accuracy))

            # visualize output after SKIP_STEP
            if (index+1) % SKIP_STEP == 0:
                visualize_output(sess, X, params, index)

            # update the global_batch_offset  
            sess.run(global_batch_offset.assign(batch_offset))
            # plot loss and accuracy
            plt.plot(loss_all)
            plt.savefig('fig/loss.png')
            plt.close()
            plt.plot(acc_all)
            plt.savefig('fig/acc.png')
            plt.close()

            # save model after SKIP_STEP
            if (index+1) % SKIP_STEP == 0:
                saver.save(sess, CHECKPOINT_PATH+'/checkpoint', index)
                # save the loss
                f = open(LOSS_FILE, 'wb')
                pickle.dump(loss_all, f)
                f.close()
                # save accuracy
                f = open(ACC_FILE, 'wb')
                pickle.dump(acc_all, f)
                f.close()

    print('\nDone\n')


if __name__ == '__main__':
    main()