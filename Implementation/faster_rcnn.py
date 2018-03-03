### Preparation #######################################################################################################

# Import packages
import os
import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from anchors.create_anchors_tensor import *
from anchors.anchors_evaluation import *
from batch_generator import MNISTCollage
from data_generation.data_gen import *
from functools import partial
from network.layers import convolutional, fully_connected, roi_pooling
from vgg16.vgg16 import VGG16


# Set class variables

# Image generation
NUM_COLLAGES = 100
COLLAGE_SIZE = 256
MIN_NUM_IMGS = 2
MAX_NUM_IMGS = 5
REPLACEMENT = True
ALLOW_OVERHANG = False
BACKGROUND = 'black'
MIN_SCALING = 1 # original mnist images size is 28x28
MAX_SCALING = 1
SCALING_STEPS = 1
COUNTERCLOCK_ANGLE = 0
CLOCKWISE_ANGLE = 0
ROTATION_STEPS = 1

# Batch generation
BATCH_SIZE = 1

# Anchor creation and selection
IMG_SIZE = 256
VGG_FM_SIZE = 16
VGG_FM_NUM = 512
ANCHORS_SCALES = [56, 56, 56]
ANCHORS_RATIOS = [1, 1, 1]
NUM_ANCHORS = 9
LOAD_LAST_ANCHORS = True
LOWER_THRESHOLD = 0.30
UPPER_THRESHOLD = 0.70
NUM_SELECTED_ANCHORS = 256

# VGG16
CKPT_PATH = './checkpoints/'
STORE_VGG = True
RESTORE_VGG = False
VGG16_PATH = None if ~RESTORE_VGG else './checkpoints/vgg16.npy'

# RPN
REG_TO_CLS_LOSS_RATIO = 10
EPOCHS_TRAINSTEP_1 = 5
LR_RPN = 0.001
RPN_ACTFUN = tf.nn.relu
RP_PATH = 'proposals.pkl'
STORE_RPN = True
RESTORE_RPN = False
RPN_PATH = './checkpoints/rpn.ckpt'

# Fast R-CNN
ROI_FM_SIZE = 8
EPOCHS_TRAINSTEP_2 = 1
STORE_FAST = True
RESTORE_FAST = False
FAST_PATH = './checkpoints/fast.ckpt'
FINALLY_VALIDATE = True


# Generate images xor load them if they already exist with the desired properties
create_collages(num_collages=NUM_COLLAGES, collage_size=COLLAGE_SIZE, min_num_imgs=MIN_NUM_IMGS,
                max_num_imgs=MAX_NUM_IMGS, replacement=REPLACEMENT, allow_overhang=ALLOW_OVERHANG,
                background=BACKGROUND, min_scaling=MIN_SCALING, max_scaling=MAX_SCALING, scaling_steps=SCALING_STEPS,
                counterclock_angle=COUNTERCLOCK_ANGLE, clockwise_angle=CLOCKWISE_ANGLE, rotation_steps=ROTATION_STEPS)

# Create input batch generator
batcher = MNISTCollage('./data_generation')

# Create anchor tensor
anchors = create_anchors_tensor(NUM_COLLAGES, NUM_ANCHORS, IMG_SIZE, VGG_FM_SIZE, ANCHORS_SCALES, ANCHORS_RATIOS)
# shape: 4D, (batchsize, num_anchors*4, feature map height, feature map width)
# Note:
# x-positions of anchors in image are saved in the first 9 <third dim, fourth dim> for each image (<first dim>)
# y-"                                            " second 9 "                    "
# width "                                        " third 9 "                     "
# height "                                       " fourth "                      "

# Evaluate anchors and assign the nearest ground truth box to the anchors evaluated as positive
eval = partial(anchors_evaluation, batch_anchor_tensor=anchors, load_last_anchors=LOAD_LAST_ANCHORS, num_selected=NUM_SELECTED_ANCHORS,
               lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD)
train_ground_truth_tensor, train_selection_tensor = eval(imgs=batcher.train_data, labels=batcher.train_labels, filename='train_anchors')
valid_ground_truth_tensor, valid_selection_tensor = eval(imgs=batcher.valid_data, labels=batcher.valid_labels, filename='valid_anchors')
test_ground_truth_tensor, test_selection_tensor = eval(imgs=batcher.test_data, labels=batcher.test_labels, filename='test_anchors')
# These methods should return two tensors:
# First tensor: Ground truth box tensor of shape (NUM_IMGS, NUM_ANCHORS*4, FM_WIDTH, FM_HEIGHT)
# Second tensor: Selection tensor (NUM_IMGS, NUM_ANCHORS*4, FM_WIDTH, FM_HEIGHT, [ANCHOR_TYPE, MNIST_CLASS, IoU]),
#                where ANCHOR_TYPE is either positive (=1), negative (=0), neutral (=-1) or deactivated
#                (= -3) and MNIST_CLASS indicates the mnist number class of the assigned ground truth mnist image xor
#                '-2' if no ground truth box was assigned

# swap dimensions of anchor tensors to fit the shape of the predicted coordinates tensor of the RPN
# and add length 1 zero dimension
swapaxes = lambda x: np.swapaxes(np.swapaxes(x, 1, 2), 2, 3)
anchors = swapaxes(anchors).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4))
train_ground_truth_tensor = swapaxes(train_ground_truth_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4))
train_selection_tensor = swapaxes(train_selection_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3))
valid_ground_truth_tensor = swapaxes(valid_ground_truth_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4))
valid_selection_tensor = swapaxes(valid_selection_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3))
test_ground_truth_tensor = swapaxes(test_ground_truth_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4))
test_selection_tensor = swapaxes(test_selection_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3))



### Data Flow Graph Construction Phase ################################################################################

### ImageNet

with tf.variable_scope('imagenet'):
    X = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_NUM_IMGS, 7])
    vgg16 = VGG16(vgg16_npy_path=VGG16_PATH)
    vgg16.build(X)


### Region Proposal Network RPN

with tf.variable_scope('rpn'):

    with tf.name_scope('placeholders'):
        anchor_coordinates = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4])
        groundtruth_coordinates = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4])
        selection_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3])

    with tf.variable_scope('pre_heads_layer'):
        prehead_conv = convolutional(vgg16.conv5_3, [3, 3, 512, 512], 1, False, RPN_ACTFUN)
        # results in a tensor with shape (1, IMG_SIZE, IMG_SIZE, 512)

    with tf.name_scope('regression_head'):
        with tf.variable_scope('predictions'):
            predicted_coordinates = convolutional(prehead_conv, [1, 1, 512, 36], 1, False)
            # results in a tensor with shape (1, IMG_SIZE, IMG_SIZE, 36)

        with tf.variable_scope('regression_parametrization'):
            # perform boxregression parametrization by transforming the coordinates and width and height to yield the
            # predicted and true (=target) coordinate parameters used for the regression loss
            tx_xor_ty = lambda t1, t2, t3: tf.divide(tf.subtract(t1, t2), t3)
            tw_xor_th = lambda t1, t2: tf.log(tf.divide(t1, t2))
            tx = tx_xor_ty(predicted_coordinates[:, :, :, 0:9], anchor_coordinates[:, :, :, 0:9], anchor_coordinates[:, :, :, 18:27])
            ty = tx_xor_ty(predicted_coordinates[:, :, :, 9:18], anchor_coordinates[:, :, :, 9:18], anchor_coordinates[:, :, :, 27:36])
            tw = tw_xor_th(predicted_coordinates[:, :, :, 18:27], anchor_coordinates[:, :, :, 18:27])
            th = tw_xor_th(predicted_coordinates[:, :, :, 27:36], anchor_coordinates[:, :, :, 27:36])
            t_predicted = tf.concat([tx, ty, tw, th], axis=0)
            tx = tx_xor_ty(groundtruth_coordinates[:, :, :, 0:9], anchor_coordinates[:, :, :, 0:9], anchor_coordinates[:, :, :, 18:27])
            ty = tx_xor_ty(groundtruth_coordinates[:, :, :, 9:18], anchor_coordinates[:, :, :, 9:18], anchor_coordinates[:, :, :, 27:36])
            tw = tw_xor_th(groundtruth_coordinates[:, :, :, 18:27], anchor_coordinates[:, :, :, 18:27])
            th = tw_xor_th(groundtruth_coordinates[:, :, :, 27:36], anchor_coordinates[:, :, :, 27:36])
            t_target = tf.concat([tx, ty, tw, th], axis=0)
            # t_target and t_predicted should have shape (4, feature map size, feature map size, number of anchors)

            with tf.variable_scope('regression_loss'):
                def smooth_l1_loss(raw_deviations, selection_tensor):
                    # raw deviations of shape (4, 16, 16, 9)
                    # select deviations for anchors marked as positive
                    activation_value = tf.constant(1.0, dtype=tf.float32)
                    filter_plane = tf.cast(tf.equal(selection_tensor[:, :, :, :, 0], activation_value), tf.float32)
                    # filter plane shape: (1, 16, 16, 9)
                    filter_tensor = tf.tile(filter_plane, [4, 1, 1, 1])
                    filtered_tensor = tf.multiply(raw_deviations, filter_tensor)
                    filtered_tensor = tf.where(tf.is_nan(filtered_tensor), tf.zeros_like(filtered_tensor), filtered_tensor)

                    # calculate the smooth l1 loss
                    absolute_deviations = tf.abs(filtered_tensor)
                    # shape: (4, 16, 16, 9)
                    # case 1: l(x), |x| < 1
                    case1_sel_tensor = tf.less(absolute_deviations, 1)
                    # shape: (4, 16, 16, 9)
                    case1_deviations = tf.multiply(absolute_deviations, tf.cast(case1_sel_tensor, tf.float32))
                    # shape: (4, 16, 16, 9)
                    case1_output = tf.multiply(tf.square(case1_deviations), 0.5)
                    # shape: (4, 16, 16, 9)
                    # case 2: otherwise
                    case2_sel_tensor = tf.greater_equal(absolute_deviations, 1)
                    # shape: (4, 16, 16, 9)
                    case2_output = tf.subtract(absolute_deviations, 0.5)
                    # shape: (4, 16, 16, 9)
                    case2_output = tf.multiply(case2_output, tf.cast(case2_sel_tensor, tf.float32))
                    # shape: (4, 16, 16, 9)
                    smooth_anchor_losses = case1_output + case2_output
                    # shape: (4, 16, 16, 9)
                    unnormalized_reg_loss = tf.reduce_sum(smooth_anchor_losses)
                    normalized_reg_loss = tf.truediv(unnormalized_reg_loss, tf.cast((VGG_FM_SIZE ** 2) * 9, tf.float32))

                    return normalized_reg_loss

                raw_deviations = tf.subtract(t_predicted, t_target)
                rpn_reg_loss_normalized = smooth_l1_loss(raw_deviations, selection_tensor)

        with tf.variable_scope('classification_head'):
            clshead_conv1 = convolutional(prehead_conv, [1, 1, 512, NUM_ANCHORS * 2], 1, False, RPN_ACTFUN)
            # should be of shape (BATCH_SIZE, 16, 16, NUM_ANCHORS*2)

            with tf.variable_scope('classification_loss'):
                # filter logits for the 256 to be activated anchors
                logits = tf.reshape(clshead_conv1, [BATCH_SIZE * VGG_FM_SIZE * VGG_FM_SIZE * NUM_ANCHORS, 2])
                # shape: (Batch size * fm size * fm size, 2)
                reshaped_targets = tf.reshape(selection_tensor[:, :, :, :, 0],
                                              [BATCH_SIZE * VGG_FM_SIZE * VGG_FM_SIZE * NUM_ANCHORS, 1])
                # shape: (Batch size * fm size * fm size, 1)
                inclusion_idxs = tf.greater_equal(reshaped_targets, 0)
                # 256 True, rest False
                tmp2 = tf.boolean_mask(tf.reshape(logits[:, 0], [tf.shape(logits)[0], 1]), inclusion_idxs)
                tmp3 = tf.boolean_mask(tf.reshape(logits[:, 1], [tf.shape(logits)[0], 1]), inclusion_idxs)
                logits_filtered = tf.concat(
                    [tf.reshape(tmp2, [tf.shape(tmp2)[0], 1]), tf.reshape(tmp3, [tf.shape(tmp3)[0], 1])], axis=1)
                # filter label entries according to the filtered logits
                sampled_targets = tf.reshape(tf.boolean_mask(reshaped_targets, inclusion_idxs), [tf.shape(tmp3)[0], 1])
                tmp4 = tf.ones_like(sampled_targets)
                idx = tf.equal(sampled_targets, 1)
                idxi = tf.not_equal(sampled_targets, 1)
                targets_filtered = tf.concat(
                    [tf.multiply(tmp4, tf.cast(idxi, tf.float32)), tf.multiply(tmp4, tf.cast(idx, tf.float32))], axis=1)

                # calculate the cross entropy loss
                rpn_cls_loss = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(labels=targets_filtered, logits=logits_filtered))
                rpn_cls_loss_normalized = tf.truediv(rpn_cls_loss, tf.cast(NUM_SELECTED_ANCHORS, tf.float32))

        with tf.name_scope('overall_loss'):
            overall_loss = rpn_cls_loss_normalized + REG_TO_CLS_LOSS_RATIO * rpn_reg_loss_normalized

        with tf.variable_scope('regularization'):
            # Collect all weights of the different variable scopes
            # W1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv1_1/')
            # W2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv1_2/')
            # W3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv2_1/')
            # W4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv2_2/')
            # W5 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv3_1/')
            # W6 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv3_2/')
            # W7 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv3_3/')
            # W8 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv4_1/')
            # W9 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv4_2/')
            # W10 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv4_3/')
            # W11 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv5_1/')
            # W12 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv5_2/')
            # W13 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_VGG16/conv5_3/')
            # W14 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rpn/pre_heads_layer/')
            # W15 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rpn/predictions/')
            # W16 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rpn/classification_head/')
            # W17 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc6/')
            # W18 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc7/')
            # W19 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cls_fc/')
            # W20 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reg_fc/')
            # W21 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cls_out/')

            # TODO: Check whether correct and filter out biases and other variables (e.g. batch normalization)

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            # variables_names = [v.name for v in tf.trainable_variables()]
            # weights = [tf.get_variable(n) for n in variables_names]

            # L2 regularization
            # overall_loss = overall_loss + tf.sqrt(tf.reduce_sum(tf.square(weights)))

        with tf.variable_scope('costs_and_optimization'):
            global_step = tf.Variable(0, trainable=False)
            boundaries = [1200, 1600]
            values = [0.001, 0.0001, 0.000005]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            # rpn_train_op = tf.train.AdamOptimizer(learning_rate).minimize(overall_loss, global_step=global_step)
            rpn_train_op = tf.train.AdamOptimizer(LR_RPN).minimize(overall_loss)

            # regularizer = tf.nn.l2_loss(weights)
            # loss = tf.reduce_mean(loss + beta * regularizer)


### Fast R-CNN

with tf.variable_scope('fast_rcnn'):

    bbox = tf.placeholder(tf.int64, [4])

    with tf.variable_scope('roi_pooling'):
        pool5 = roi_pooling(vgg16.conv5_3, bbox, [ROI_FM_SIZE, ROI_FM_SIZE])

    with tf.variable_scope('layer_6'):
        fc6 = fully_connected(tf.reshape(pool5, [-1, np.prod(pool5.shape[1:])]), 4096, False, tf.nn.relu)

    with tf.variable_scope('layer_7'):
        fc7 = fully_connected(fc6, 1024, False, tf.nn.relu)

    with tf.variable_scope('bbox_pred'):
        bbox_pred = fully_connected(fc7, 40, False, tf.nn.relu)
        # TODO: Implement loss for regression

    with tf.variable_scope('cls_score'):
        cls_score = fully_connected(fc7, 10, False, tf.nn.relu)

    with tf.variable_scope('cls_prob'):
        cls_prob = fully_connected(cls_score, 10, False, None)
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(Y[:, 0, 0], tf.int64), logits=cls_prob))


### Model saving nodes

with tf.name_scope('model_savers'):
    vgg_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='imagenet'))
    rpn_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rpn'))
    fast_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fast_rcnn'))

### Model initialization nodes

with tf.name_scope('model_initializers'):
    init = tf.global_variables_initializer()
    vgg_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='imagenet'))
    rpn_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rpn'))
    fast_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fast_rcnn'))


### Execution Phase ###################################################################################################

if __name__ == "__main__":
#
    with tf.Session() as sess:

        # Initialize xor restore the required sub-models
        restore_xor_init = lambda restore, saver, path, ini: saver.restore(sess, path) if restore else sess.run(ini)
        restore_xor_init(RESTORE_RPN, rpn_saver, RPN_PATH, rpn_init)
        restore_xor_init(RESTORE_FAST, fast_saver, FAST_PATH, fast_init)
        if ~RESTORE_VGG:
            sess.run(vgg_init)

        #train_writer = tf.summary.FileWriter("./summaries/train", tf.get_default_graph())
        iter = 0
        proposals = []

        for epoch in range(EPOCHS_TRAINSTEP_1):
            for X_batch, Y_batch, first, last in batcher.get_batch(BATCH_SIZE):
                if BATCH_SIZE == 1:
                    _, rp, lr, lc, ol = sess.run(
                        [rpn_train_op, predicted_coordinates, rpn_reg_loss_normalized, rpn_cls_loss_normalized, overall_loss],
                        feed_dict={X: X_batch,
                                   Y: Y_batch,
                                   anchor_coordinates: anchors[first],
                                   groundtruth_coordinates: train_ground_truth_tensor[first],
                                   selection_tensor: train_selection_tensor[first]}
                    )

                    if epoch + 1 == EPOCHS_TRAINSTEP_1:
                        proposals.append(rp)

                    if iter % 10 == 0:
                        print('iteration:', iter)#, 'reg loss:', lr, 'cls loss:', lc, 'overall loss:', ol)
                    iter += 1


                with open(RP_PATH, 'wb') as file:
                    pickle.dump(proposals, file)


                for epoch in range(EPOCHS_TRAINSTEP_2):
                    for X_batch, Y_batch, first, last in batcher.get_batch(BATCH_SIZE):
                        for i, j, k in np.ndindex(16, 16, 9):
                            if train_selection_tensor[first][:, i, j, k, 0] == 1:
                                proposal = np.zeros(4)
                                proposal[0] = proposals[first][:, i, j, k]
                                proposal[1] = proposals[first][:, i, j, 9 + k]
                                proposal[2] = proposals[first][:, i, j, 18 + k]
                                proposal[3] = proposals[first][:, i, j, 27 + k]

                                out = sess.run(pool5, feed_dict={X: X_batch, bbox: proposal})
                                print(out.shape)

                storer = lambda boolean, saver, filename: saver.save(sess, CKPT_PATH + filename) if boolean else None
                storer(STORE_RPN, rpn_saver, 'rpn.ckpt')
                storer(STORE_FAST, fast_saver, 'fast.ckpt')
                if STORE_VGG:
                    vgg16.save_npy(sess, CKPT_PATH + 'vgg16.npy')



                # # Validation
                # for f in range(len(batcher.valid_data)):
                #     X_batch = batcher.valid_data[f]
                #     Y_batch = batcher.valid_labels[f]
                #     vlr, vlc, vol = sess.run([rpn_reg_loss_normalized, rpn_cls_loss_normalized, overall_loss,
                #                               predicted_coordinates, clshead_conv1],
                #                              feed_dict={X: np.array(X_batch).reshape((1, 256, 256, 3)),
                #                                         Y: np.array(Y_batch).reshape((1, 256, 256, 3)),
                #                                         anchor_coordinates: anchors[f],
                #                                         groundtruth_coordinates: valid_ground_truth_tensor[f],
                #                                         selection_tensor: valid_selection_tensor[f]})