### Preparation #######################################################################################################

# Import packages
import os
import pdb

import numpy as np
import tensorflow as tf

from anchors.create_anchors_tensor import *
from anchors.anchors_evaluation import *
from batch_generator import MNISTCollage
from data_generation.data_gen import *
from network.layers import convolutional, fully_connected
from vgg16.vgg16 import VGG16


# Set class variables

# Image generation class variables
NUM_COLLAGES = 100
COLLAGE_SIZE = 256
MIN_NUM_IMGS = 1
MAX_NUM_IMGS = 3
REPLACEMENT = True
ALLOW_OVERHANG = True
BACKGROUND = 'black'
MIN_SCALING = 0.51
MAX_SCALING = 1.5
SCALING_STEPS = 2
COUNTERCLOCK_ANGLE = 0
CLOCKWISE_ANGLE = 0
ROTATION_STEPS = 2

# Batch generator class variable
BATCH_SIZE = 1

# Anchor creation and selection class variables
IMG_SIZE = 256
VGG_FM_SIZE = 16
VGG_FM_NUM = 512
ANCHORS_SCALES = [42, 28, 14]
ANCHORS_RATIOS = [1.75, 1, 0.40]
NUM_ANCHORS = 9
LOAD_LAST_ANCHORS = True
NUM_SELECTED_ANCHORS = 256
ROI_FM_SIZE = 8
NUM_CLASSES = 10

EPOCHS_TRAINSTEP1 = 2


# Generate images xor load them if they already exist with the desired properties
create_collages(num_collages=NUM_COLLAGES, collage_size=COLLAGE_SIZE, min_num_imgs=MIN_NUM_IMGS,
                max_num_imgs=MAX_NUM_IMGS, replacement=REPLACEMENT, allow_overhang=ALLOW_OVERHANG,
                background=BACKGROUND, min_scaling=MIN_SCALING, max_scaling=MAX_SCALING, scaling_steps=SCALING_STEPS,
                counterclock_angle=COUNTERCLOCK_ANGLE, clockwise_angle=CLOCKWISE_ANGLE, rotation_steps=ROTATION_STEPS)

# Create input batch generator
batcher = MNISTCollage('./data_generation')
train_labels = batcher.train_labels
valid_labels = batcher.valid_labels
test_labels  = batcher.test_labels

# For debugging: Examine image and labels of batches
#for x,y in batcher.get_batch(4):
#    pdb.set_trace()


# Create anchor tensor

anchors = create_anchors_tensor(NUM_COLLAGES, NUM_ANCHORS, IMG_SIZE, VGG_FM_SIZE, ANCHORS_SCALES, ANCHORS_RATIOS)
# shape: 4D, (batchsize, num_anchors*4, feature map height, feature map width)
# Note:
# x-positions of anchors in image are saved in the first 9 <third dim, fourth dim> for each image (<first dim>)
# y-"                                            " second 9 "                    "
# width "                                        " third 9 "                     "
# height "                                       " fourth "                      "


# Evaluate anchors and assign the nearest ground truth box to the anchors evaluated as positive
train_ground_truth_tensor, train_selection_tensor = anchors_evaluation(batch_anchor_tensor=anchors,
                                                                       imgs=batcher.train_data, labels=train_labels,
                                                                       load_last_anchors=LOAD_LAST_ANCHORS,
                                                                       filename='train_anchors',
                                                                       num_selected=NUM_SELECTED_ANCHORS)
valid_ground_truth_tensor, valid_selection_tensor = anchors_evaluation(batch_anchor_tensor=anchors,
                                                                       imgs=batcher.valid_data, labels=valid_labels,
                                                                       load_last_anchors=LOAD_LAST_ANCHORS,
                                                                       filename='valid_anchors',
                                                                       num_selected=NUM_SELECTED_ANCHORS)
test_ground_truth_tensor, test_selection_tensor = anchors_evaluation(batch_anchor_tensor=anchors,
                                                                     imgs=batcher.test_data, labels=test_labels,
                                                                     load_last_anchors=LOAD_LAST_ANCHORS,
                                                                     filename='test_anchors',
                                                                     num_selected=NUM_SELECTED_ANCHORS)
# These methods should return two tensors:
# First tensor: Ground truth box tensor of shape (NUM_IMGS, NUM_ANCHORS*4, FM_WIDTH, FM_HEIGHT)
# Second tensor: Selection tensor (NUM_IMGS, NUM_ANCHORS*4, FM_WIDTH, FM_HEIGHT, [ANCHOR_TYPE, MNIST_CLASS, IoU]),
#                where ANCHOR_TYPE is either positive (=1), negative (=0), neutral (=-1) or deactivated
#                (= -3) and MNIST_CLASS indicates the mnist number class of the assigned ground truth mnist image xor
#                '-2' if no ground truth box was assigned


# TODO: Problem --> only very few anchors show ioU > 0.7 --> possible causes:
# TODO: 1. inadequate scale of mnist images on collages, 2. inadequate scale of anchors, 3. too coarse feature map

# TODO: NMS

# Debugging the ground truth and selection tensor II
#import matplotlib.pyplot as plt
#for img in range(5):
#    plt.imshow(batcher.train_data[img,:,:])
#    plt.show()
#    print(train_labels[img])
#    for j in range(2):
#        print(train_ground_truth_tensor[img,j,:,:])
#        print(train_selection_tensor[img,j,:,:,0])
#    pdb.set_trace()



# TODO: Deep debugging of ground truth and selection tensors
pdb.set_trace()



### Data Flow Graph Construction Phase ################################################################################


#def create_graph():
#   """ Creates a graph from saved GraphDef file and returns a saver. """
#   with tf.gfile.FastGFile(os.path.join("imagenet", "classify_image_graph_def.pb"), 'rb') as file:
#       graph_def = tf.GraphDef()
#       graph_def.ParseFromString(file.read())
#       tf.import_graph_def(graph_def, name='')


### Region Proposal Network RPN

with tf.variable_scope('rpn'):

    with tf.name_scope('placeholders'):
        X = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, VGG_FM_NUM])
        Y = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_NUM_IMGS, 7])
        # TODO: Might be sufficient to just hand over the classes of the single mnist images
        anchor_coordinates = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_ANCHORS*4, VGG_FM_SIZE, VGG_FM_SIZE])
        groundtruth_coordinates = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_ANCHORS*4, VGG_FM_SIZE, VGG_FM_SIZE])
        selection_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_ANCHORS*4, VGG_FM_SIZE, VGG_FM_SIZE])

    with tf.name_scope('anchor_tensors_dimension_rearrangement'):
        # swap dimensions of anchor, groundtruth box and selection tensors so they fit the predicted coordinate tensor
        anchor_coordinates = tf.transpose(anchor_coordinates, [0, 2, 3, 1])
        groundtruth_coordinates = tf.transpose(groundtruth_coordinates, [0, 2, 3, 1])
        selection_tensor = tf.transpose(selection_tensor, [0, 2, 3, 1])

    with tf.variable_scope('pre_heads_layers'):
        prehead_conv = convolutional(X, [3, 3, 512, 512], 1, True, tf.nn.relu)
        # results in a tensor with shape (1, IMG_SIZE, IMG_SIZE, 512)

    with tf.variable_scope('regression_head'):
        predicted_coordinates = convolutional(prehead_conv, [1, 1, 512, 36], 1, True)
        # results in a tensor with shape (1, IMG_SIZE, IMG_SIZE, 36
        predicted_x = predicted_coordinates[:, :, :, 0:9]
        predicted_y = predicted_coordinates[:, :, :, 9:18]
        predicted_w = predicted_coordinates[:, :, :, 18:27]
        predicted_h = predicted_coordinates[:, :, :, 27:36]

        anchors_x = anchor_coordinates[:, :, :, 0:9]
        anchors_y = anchor_coordinates[:, :, :, 9:18]
        anchors_w = anchor_coordinates[:, :, :, 18:27]
        anchors_h = anchor_coordinates[:, :, :, 27:36]

        t_predicted_x = tf.divide(tf.subtract(predicted_x, anchors_x), anchors_w)
        t_predicted_y = tf.divide(tf.subtract(predicted_y, anchors_y), anchors_h)
        t_predicted_w = tf.log(tf.divide(predicted_w, anchors_w))
        t_predicted_h = tf.log(tf.divide(predicted_h, anchors_h))
        t_target_x = tf.divide(tf.subtract(true_x, anchors_x), anchors_w)
        t_target_y = tf.divide(tf.subtract(true_y, anchors_y), anchors_h)
        t_target_w = tf.log(tf.divide(true_w, anchors_w))
        t_target_h = tf.log(tf.divide(true_h, anchors_h))



        #conv1_transposed = tf.transpose(conv1, [0,2,3,1])
        #N, W, H, K = tf.shape(conv1_transposed)
        #conv1_reshaped = tf.reshape(conv1_transposed, [int((N*W*H*K)/4), 4])
        #prediction = 0

    with tf.variable_scope('classification_head'):
        clshead_conv1 = convolutional(prehead_conv, [1, 1, 512, 32], 1, True, tf.nn.relu)

    with tf.variable_scope('costs_and_optimization'):
        pass


with tf.name_scope('fast_rcnn'):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_NUM * ROI_FM_SIZE**2])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])

    with tf.variable_scope('fc6'):
        fc6 = fully_connected(x, 4096, False, tf.nn.relu)
    with tf.variable_scope('fc7'):
        fc7 = fully_connected(fc6, 4096, False, tf.nn.relu)
    with tf.variable_scope('cls_fc'):
        cls_fc = fully_connected(fc7, 10, False, tf.nn.relu)
    with tf.variable_scope('reg_fc'):
        reg_fc = fully_connected(fc7, 40, False, tf.nn.relu)
    with tf.variable_scope('cls_out'):
        cls_out = fully_connected(cls_fc, 10, False, None)
        cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=cls_fc))
    # TODO: Implement loss for regression


### Execution Phase ###################################################################################################


if __name__ == "__main__":

    # Load pretrained VGG16 and get handle on input placeholder
    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    vgg16 = VGG16()
    vgg16.build(inputs)

    # read out last pooling layer
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        #train_writer = tf.summary.FileWriter("./summaries/train", tf.get_default_graph())

        for epoch in range(EPOCHS_TRAINSTEP1):

            for X_batch, Y_batch in batcher.get_batch(BATCH_SIZE):

                result_tensor = sess.graph.get_tensor_by_name('conv5_3/Relu:0')
                vgg16_conv5_3_relu = sess.run(result_tensor, feed_dict={inputs: X_batch})
                print(vgg16_conv5_3_relu.shape)
                # output of VGG16 will be of shape (BATCHSIZE, 8, 8, 512)

                _ = sess.run([predicted_h], feed_dict={X: vgg16_conv5_3_relu,
                                                            Y: Y_batch})
                print(_[0].shape)

                # TODO: Create ground truth boxregression tensor (here, after batching): (BATCHSIZE, NUM_ANCHORS*4, W, H)
                # TODO: ... how to implement it? As a sparse tensor? Alternatives?
                # 1. For every image positive, neutral and negative anchors must be identified
                # 2. For the positive anchors strongest related ground truth box (coordinates) must be
                #    determined
                #    -> Idea: Can be done BEFORE DFG / TF
                # 3. Regression loss is calculated only between positive anchors and strongest related
                #    ground truth box AND ADDITIONALLY A SUBSAMPLE OF THE ANCHORS IS DRAWN, SEE PAPER
                #    -> Tensor approach can be applied, but negative anchor loss must be zero weighted
                #       or better: not computed at all
                # 4. NMS etc. (, filtering out too small boxes, selecting top N boxes) seem to come
                #    after bbox regression and classification


                # TODO: Create predicted boxregression tensor (inside the graph?)

                #tmp = sess.run([prehead_conv], feed_dict={X: X_batch,
                #                                    Y: Y_batch,
                #                                    values_anchors: anchors,
                #                                    })
                #print(tmp)

