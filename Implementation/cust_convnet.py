''' simple convolutional network designed for recognizing single mnist numbers of different rotation angles and
    scales on the collage frame '''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from network.layers import convolutional, fully_connected
import pickle
import os
import pdb

BATCHSIZE = 5
LEARNING_RATE = 0.001
EPOCHS = 1

def loader():
    train, valid, test = [], [], []
    for file in os.listdir('cust_cnn_data'):
        if 'collage' in file or 'targets' in file:
            if 'train' in file:
                with open('cust_cnn_data/'+file, 'rb') as f:
                    train.append(pickle.load(f))
            if 'valid' in file:
                with open('cust_cnn_data/'+file, 'rb') as f:
                    valid.append(pickle.load(f))
    return [train[0], train[1], valid[0], valid[1]]


# Enshure right data types and dimensionality of data
X_train, y_train, X_valid, y_valid = loader()
X_train = np.array(X_train)
X_valid = np.array(X_valid)
y_train = np.array(y_train)
y_train = np.array([l[0][0] for l in y_train])
y_valid = np.array([l[0][0] for l in y_valid])

for i in range(len(X_train)):
    plt.imshow(X_train[i])
    print(y_train[i])
    plt.show()
    plt.imshow(X_valid[i])
    print(y_valid[i])
    plt.show()


def one_hot(data, cols=10):
    tmp = np.zeros((data.shape[0],10))
    for e in range(data.shape[0]):
        tmp[e, int(data[e])] = 1
    return tmp
y_train = one_hot(y_train)
y_valid = one_hot(y_valid)

with tf.name_scope('placeholders'):
    X = tf.placeholder(dtype=tf.float32, shape=[BATCHSIZE, 128, 128, 1], name='img')
    y = tf.placeholder(dtype=tf.int32, shape=[BATCHSIZE, 10], name='label')

with tf.name_scope('conv_layers'):
    with tf.variable_scope('conv1'):
        conv1 = convolutional(X,     [3, 3, 1, 32], 1, True, tf.nn.relu)
        pool1 = tf.nn.avg_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    with tf.variable_scope('conv2'):
        conv2 = convolutional(pool1, [3, 3, 32,64], 1, True, tf.nn.relu)
        pool2 = tf.nn.avg_pool(conv2, [1, 3, 3, 1], [1, 3, 3, 1], 'SAME')
    with tf.variable_scope('conv3'):
        conv3 = convolutional(conv2, [3, 3,64,128], 1, True, tf.nn.relu)
    with tf.variable_scope('conv4'):
        conv4 = convolutional(conv3, [3,3,128,256], 1, True, tf.nn.relu)
    with tf.variable_scope('conv5'):
        conv5 = convolutional(conv4, [3,3,256,512], 1, True, tf.nn.relu)
        pool3 = tf.nn.avg_pool(conv5, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

with tf.name_scope('ff_layers'):
    flattened = tf.reshape(pool2, [BATCHSIZE, -1])# shape supposed to be: (batchsize, k)
    with tf.variable_scope('ff1'):
        ff1 = fully_connected(flattened, 1024, False, tf.nn.relu)
    with tf.variable_scope('ff2'):
        ff1 = fully_connected(flattened,  512, False, tf.nn.relu)
    logits = fully_connected(flattened,    10, False, tf.nn.relu)

with tf.name_scope('costs_and_optimization'):
    costs = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(y,tf.float32), logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y,tf.float32),logits), tf.float32))
    optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(costs)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    cos, acc, cosv, accv = [], [], [], []
    for e in range(EPOCHS):

        for b in range(len(X_train)//BATCHSIZE):
            r_idxs_t = np.random.choice(np.arange(0, X_train.shape[0]), BATCHSIZE)
            r_idxs_v = np.random.choice(np.arange(0, X_train.shape[0]), BATCHSIZE)
            xt = np.array(X_train[r_idxs_t]).reshape((BATCHSIZE,X_train.shape[1],X_train.shape[2],1))
            xv = np.array(X_valid[r_idxs_v]).reshape((BATCHSIZE,X_valid.shape[1],X_valid.shape[2],1))
            yt = y_train[r_idxs_t]
            yv = y_valid[r_idxs_v]
            _, cos_, acc_ = sess.run([costs, accuracy, optimize], feed_dict={X: xt, y: yt})
            cos.append(cos_)
            acc.append(acc_)
            cosv_, accv_ = sess.run([costs, accuracy], feed_dict={X: xv, y: yv})
            cosv.append(cos_)
            accv.append(acc_)
            print('train costs:',cos_,', accuracy:',acc_,', valid. costs:',cosv_,', accuracy:',accv_)

    saver.save(sess, "/custom_convnet_checkpoints/model.ckpt")

for c, a in zip(cos, acc):
    plt.subplot(2,2,1)
    plt.plot(c)
    plt.ylabel('cross entropy')
    plt.subplot(2,2,2)
    plt.plot(a)
    plt.ylabel('accuracy')
    plt.subplot(2,2,3)
    plt.plot(c)
    plt.xlabel('iteration')
    plt.ylabel('validation cross entropy')
    plt.subplot(2,2,4)
    plt.plot(a)
    plt.xlabel('iteration')
    plt.ylabel('validation accuracy')
    plt.show()
