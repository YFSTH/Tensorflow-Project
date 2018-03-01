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
from network.layers import convolutional, fully_connected, roi_pooling
from vgg16.vgg16 import VGG16
from Optimizers.AMSGrad import *


# Set class variables

# Image generation class variables
NUM_COLLAGES = 100
COLLAGE_SIZE = 256
MIN_NUM_IMGS = 2
MAX_NUM_IMGS = 5
REPLACEMENT = True
ALLOW_OVERHANG = False
BACKGROUND = 'black'
MIN_SCALING = 2.0 # original mnist images size is 28x28
MAX_SCALING = 2.0
SCALING_STEPS = 1
COUNTERCLOCK_ANGLE = 0
CLOCKWISE_ANGLE = 0
ROTATION_STEPS = 1

# Batch generator class variable
BATCH_SIZE = 1

# Anchor creation and selection class variables
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

# Fast R-CNN class variables
ROI_FM_SIZE = 8
NUM_CLASSES = 10

# RPN
REG_TO_CLS_LOSS_RATIO = 10
EPOCHS_TRAINSTEP1 = 1
LR_RPN = 0.001
RPN_ACTFUN = tf.nn.elu

# TODO: Employ regularization

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
                                                                       num_selected=NUM_SELECTED_ANCHORS,
                                                                       lower_threshold=LOWER_THRESHOLD,
                                                                       upper_threshold=UPPER_THRESHOLD)
valid_ground_truth_tensor, valid_selection_tensor = anchors_evaluation(batch_anchor_tensor=anchors,
                                                                       imgs=batcher.valid_data, labels=valid_labels,
                                                                       load_last_anchors=LOAD_LAST_ANCHORS,
                                                                       filename='valid_anchors',
                                                                       num_selected=NUM_SELECTED_ANCHORS,
                                                                       lower_threshold=LOWER_THRESHOLD,
                                                                     upper_threshold=UPPER_THRESHOLD)
test_ground_truth_tensor, test_selection_tensor = anchors_evaluation(batch_anchor_tensor=anchors,
                                                                     imgs=batcher.test_data, labels=test_labels,
                                                                     load_last_anchors=LOAD_LAST_ANCHORS,
                                                                     filename='test_anchors',
                                                                     num_selected=NUM_SELECTED_ANCHORS,
                                                                     lower_threshold=LOWER_THRESHOLD,
                                                                     upper_threshold=UPPER_THRESHOLD)
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

#pdb.set_trace()


### Data Flow Graph Construction Phase ################################################################################

### ImageNet

X = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
vgg16 = VGG16()
vgg16.build(X)


### Region Proposal Network RPN

with tf.variable_scope('rpn'):

    with tf.name_scope('placeholders'):

        #X = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, VGG_FM_NUM])
        Y = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_NUM_IMGS, 7])
        # TODO: Might be sufficient to just hand over the classes of the single mnist images
        anchor_coordinates = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4])
        groundtruth_coordinates = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4])
        selection_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3])

    with tf.variable_scope('pre_heads_layer'):
        #prehead_conv = convolutional(X, [3, 3, 512, 512], 1, False, RPN_ACTFUN)
        prehead_conv = convolutional(vgg16.conv5_3, [3, 3, 512, 512], 1, False, tf.nn.relu)
        # results in a tensor with shape (1, IMG_SIZE, IMG_SIZE, 512)

    with tf.name_scope('regression_head'):

        with tf.variable_scope('predictions'):

            predicted_coordinates = convolutional(prehead_conv, [1, 1, 512, 36], 1, False)
            # results in a tensor with shape (1, IMG_SIZE, IMG_SIZE, 36)

        with tf.variable_scope('regression_parametrization'):

            # perform boxregression parametrization by transforming the coordinates and width and height to yield the
            # predicted and true (=target) coordinate parameters used for the regression loss
            anchors_x = anchor_coordinates[:, :, :, 0:9]
            anchors_y = anchor_coordinates[:, :, :, 9:18]
            anchors_w = anchor_coordinates[:, :, :, 18:27]
            anchors_h = anchor_coordinates[:, :, :, 27:36]
            predicted_x = predicted_coordinates[:, :, :, 0:9]
            predicted_y = predicted_coordinates[:, :, :, 9:18]
            predicted_w = predicted_coordinates[:, :, :, 18:27]
            predicted_h = predicted_coordinates[:, :, :, 27:36]
            target_x = groundtruth_coordinates[:, :, :, 0:9]
            target_y = groundtruth_coordinates[:, :, :, 9:18]
            target_w = groundtruth_coordinates[:, :, :, 18:27]
            target_h = groundtruth_coordinates[:, :, :, 27:36]
            t_predicted_x = tf.truediv(tf.subtract(predicted_x, anchors_x), anchors_w)
            t_predicted_y = tf.truediv(tf.subtract(predicted_y, anchors_y), anchors_h)
            t_predicted_w = tf.log(tf.truediv(predicted_w, anchors_w))
            t_predicted_h = tf.log(tf.truediv(predicted_h, anchors_h))
            t_predicted = tf.concat([t_predicted_x, t_predicted_y, t_predicted_w, t_predicted_h], 0)
            t_target_x = tf.truediv(tf.subtract(target_x, anchors_x), anchors_w)
            t_target_y = tf.truediv(tf.subtract(target_y, anchors_y), anchors_h)
            t_target_w = tf.log(tf.truediv(target_w, anchors_w))
            t_target_h = tf.log(tf.truediv(target_h, anchors_h))
            t_target = tf.concat([t_target_x, t_target_y, t_target_w, t_target_h], 0)
            # t_target and t_predicted should have shape (4, feature map size, feature map size, number of anchors)

        with tf.variable_scope('regression_loss'):
            def smooth_l1_loss(raw_deviations, selection_tensor):
                # raw deviations of shape (4, 16, 16, 9)
                # select deviations for anchors marked as positive
                activation_value = tf.constant(1.0, dtype=tf.float32)
                filter_plane = tf.cast(tf.equal(selection_tensor[:, :, :, :, 0], activation_value), tf.float32)

                # remove nans from tensor to enable aggregating calculations
                # filter_plane = tf.where(tf.is_nan(filter_plane), tf.zeros_like(filter_plane),
                #                          filter_plane)

                # filter plane shape: (1, 16, 16, 9)
                # auf ebene 1 ein positive value auf ebene 7 zwei positive values
                filter_tensor = tf.tile(filter_plane, [4, 1, 1, 1])

                filtered_tensor = tf.multiply(raw_deviations, filter_tensor)

                filtered_tensor = tf.where(tf.is_nan(filtered_tensor), tf.zeros_like(filtered_tensor),
                                           filtered_tensor)

                # calculate the smooth l1 loss

                # sum up deviations for the four coordinates per anchor

                absolute_deviations = tf.abs(filtered_tensor)
                #absolute_deviations = tf.reduce_sum(tf.abs(filtered_tensor), 0)
                # shape: (4, 16, 16, 9)

                # TODO: Überall für nans korrigieren

                # absolute deviations
                #absolute_deviations = tf.abs(summed_deviations)

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

            #def euclidian_loss(raw_deviations, selection_tensor):
            # TODO: erroneous
            #    activation_value = tf.constant(1.0, dtype=tf.float32)
            #    filter_plane = tf.cast(tf.equal(selection_tensor[:, :, :, :, 0], activation_value), tf.float32)
            #    filter_tensor = tf.tile(filter_plane, [4, 1, 1, 1])
            #    filtered_tensor = tf.multiply(raw_deviations, filter_tensor)
            #    filtered_tensor = tf.where(tf.is_nan(filtered_tensor), tf.zeros_like(filtered_tensor),
            #                               filtered_tensor)
            #    return tf.divide(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(filtered_tensor), axis=0))), (VGG_FM_SIZE ** 2) * 9)

            def l1_loss(raw_deviations, selection_tensor):
                activation_value = tf.constant(1.0, dtype=tf.float32)
                filter_plane = tf.cast(tf.equal(selection_tensor[:, :, :, :, 0], activation_value), tf.float32)
                filter_tensor = tf.tile(filter_plane, [4, 1, 1, 1])
                filtered_tensor = tf.abs(tf.multiply(raw_deviations, filter_tensor))
                filtered_tensor = tf.where(tf.is_nan(filtered_tensor), tf.zeros_like(filtered_tensor), filtered_tensor)
                return tf.divide(tf.reduce_sum(filtered_tensor), tf.cast((VGG_FM_SIZE ** 2) * 9, tf.float32))

            raw_deviations = tf.subtract(t_predicted, t_target)
            rpn_reg_loss_normalized = smooth_l1_loss(raw_deviations, selection_tensor)
            #rpn_reg_loss_normalized = l1_loss(raw_deviations, selection_tensor)
            # TODO: Test and debug euclidian loss

            # TODO: Problem: Box regression inaccurate, region proposals won´t sufficiently fit the ground truth boxes
            # TODO: Possible solutions: 1. add convlayer tween prehead and reg / class, 2. trainable VGG, 3. other error
            # TODO:                     ... func, errors somewhere, more weighting to regloss

            # TODO: KOMPLETT DURCHRECHNEN, WURDEN IRGENDWO WERTE VERTAUSCHT?

            # TODO: Why don´t hit all 9 tensors? -> cause > .70

            # TODO: Predictions nähern sich Nutzbarkeit => Finetuning -> AMSGrad, AdaGrad, RMSProp, Nesterov & Vanilla Momentum

    with tf.variable_scope('classification_head'):
        clshead_conv1 = convolutional(prehead_conv, [1, 1, 512, NUM_ANCHORS*2], 1, False, RPN_ACTFUN)
        # should be of shape (BATCH_SIZE, 16, 16, NUM_ANCHORS*2)

        with tf.variable_scope('classification_loss'):

            # filter logits for the 256 to be activated anchors
            logits = tf.reshape(clshead_conv1, [BATCH_SIZE*VGG_FM_SIZE*VGG_FM_SIZE*NUM_ANCHORS, 2])
            # shape: (Batch size * fm size * fm size, 2)
            reshaped_targets = tf.reshape(selection_tensor[:, :, :, :, 0], [BATCH_SIZE*VGG_FM_SIZE*VGG_FM_SIZE*NUM_ANCHORS, 1])
            # shape: (Batch size * fm size * fm size, 1)

            inclusion_idxs = tf.greater_equal(reshaped_targets, 0)
            # TODO: Hier wurde auf andere Methode zurückgegriffen, als oben: boolean mask statt Multiplikation, dann kein
            # TODO: ... reshape notwendig
            # 256 True, rest False

            tmp2 = tf.boolean_mask(tf.reshape(logits[:,0], [tf.shape(logits)[0],1]), inclusion_idxs)
            tmp3 = tf.boolean_mask(tf.reshape(logits[:,1], [tf.shape(logits)[0],1]), inclusion_idxs)
            logits_filtered = tf.concat([tf.reshape(tmp2, [tf.shape(tmp2)[0], 1]),tf.reshape(tmp3, [tf.shape(tmp3)[0], 1])], axis=1)

            # filter label entries according to the filtered logits
            sampled_targets = tf.reshape(tf.boolean_mask(reshaped_targets, inclusion_idxs), [tf.shape(tmp3)[0], 1])
            tmp4 = tf.ones_like(sampled_targets)
            idx = tf.equal(sampled_targets, 1)
            idxi = tf.not_equal(sampled_targets, 1)
            targets_filtered = tf.concat([tf.multiply(tmp4, tf.cast(idxi, tf.float32)), tf.multiply(tmp4, tf.cast(idx, tf.float32))], axis=1)

            # calculate the cross entropy loss
            rpn_cls_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=targets_filtered, logits=logits_filtered))
            rpn_cls_loss_normalized = tf.truediv(rpn_cls_loss, tf.cast(NUM_SELECTED_ANCHORS, tf.float32))

    with tf.name_scope('overall_loss'):
        overall_loss = rpn_cls_loss_normalized + REG_TO_CLS_LOSS_RATIO * rpn_reg_loss_normalized

    with tf.variable_scope('costs_and_optimization'):
        global_step = tf.Variable(0, trainable=False)
        boundaries = [1200, 1600]
        values = [0.001, 0.0001, 0.000005]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        #rpn_train_op = tf.train.AdamOptimizer(LR_RPN, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(overall_loss)
        #rpn_train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(overall_loss, global_step=global_step)
        #rpn_train_op = tf.train.AdamOptimizer(learning_rate).minimize(overall_loss, global_step=global_step)
        #rpn_train_op = tf.train.AdamOptimizer(LR_RPN).minimize(overall_loss)
        AMSGrad = AMSGrad(learning_rate)
        rpn_train_op = AMSGrad.minimize(overall_loss, global_step=global_step)

        #regularizer = tf.nn.l2_loss(weights)
        #loss = tf.reduce_mean(loss + beta * regularizer)

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

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        #train_writer = tf.summary.FileWriter("./summaries/train", tf.get_default_graph())

        rp = None
        cp = None
        x_b = None
        y_b = None
        vx_b = None
        vy_b = None
        vrpreds = None
        f = None
        l = None
        tx_ = None
        ty_ = None
        tw_ = None
        th_ = None
        px_ = None
        py_ = None
        pw_ = None
        ph_ = None
        ttgs = None
        tpx_ = None
        tpy_ = None
        tpw_ = None
        tph_ = None
        tarx_ = None
        tary_ = None
        tarw_ = None
        tarh_ = None

        iter = 0
        
        lr_list = []
        lc_list = []
        oa_list = []
        vlr_list = []
        vlc_list = []
        voa_list = []

        for epoch in range(EPOCHS_TRAINSTEP1):

            for X_batch, Y_batch, first, last in batcher.get_batch(BATCH_SIZE):



                #result_tensor = sess.graph.get_tensor_by_name('conv5_3/Relu:0')
                #vgg16_conv5_3_relu = sess.run(result_tensor, feed_dict={X: X_batch})

                # output of VGG16 will be of shape (BATCHSIZE, 8, 8, 512)

                if BATCH_SIZE == 1:

                    _, tpreds, tpx, tpy, tpw, tph, rpreds, px, py, pw, ph, ttargs, tx, ty, tw, th, gtc, tarx, tary, tarw, tarh, cpreds, lr, lc, ol = sess.run([rpn_train_op, t_predicted, t_predicted_x, t_predicted_y, t_predicted_w, t_predicted_h,
                                                                                                               predicted_coordinates, predicted_x, predicted_y, predicted_w, predicted_h, 
                                                                                                               t_target, t_target_x, t_target_y, t_target_w, t_target_h, 
                                                                                                               groundtruth_coordinates, target_x, target_y, target_w, target_h,
                                                                                                               clshead_conv1, rpn_reg_loss_normalized, rpn_cls_loss_normalized, overall_loss], 
                                                                                                              feed_dict={X: X_batch,
                                                                                                                         Y: Y_batch,
                                                                                                                         anchor_coordinates: anchors[first],
                                                                                                                         groundtruth_coordinates: train_ground_truth_tensor[first],#.reshape((BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS)),
                                                                                                                         selection_tensor: train_selection_tensor[first]})#..reshape((BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3))})

                    vlr, vlc, vol, vrpreds = sess.run([rpn_reg_loss_normalized, rpn_cls_loss_normalized, overall_loss, predicted_coordinates],
                                                  feed_dict={X: np.array(batcher.valid_data[first]).reshape((1,256,256,3)),
                                                             Y: np.array(batcher.valid_labels[first]).reshape((1,5,7)),
                                                             anchor_coordinates: anchors[first],
                                                             groundtruth_coordinates: valid_ground_truth_tensor[first],
                                                             selection_tensor: valid_selection_tensor[first]})

                    f, l = first, last
                    vx_b = batcher.valid_data[first]
                    vy_b = batcher.valid_labels[first]
                    x_b = X_batch
                    y_b = Y_batch
                    rp = rpreds
                    vrp = vrpreds
                    cp = cpreds
                    tx_ = tx
                    ty_ = ty
                    tw_ = tw
                    th_ = th
                    px_ = px
                    py_ = py
                    pw_ = pw
                    ph_ = ph
                    ttgs = ttargs
                    tpx_ = tpx
                    tpy_ = tpy
                    tpw_ = tpw
                    tph_ = tph
                    tarx_ = tarx
                    tary_ = tary
                    tarw_ = tarw
                    tarh_ = tarh
                    lr_list.append(lr)
                    lc_list.append(lc)
                    oa_list.append(ol)
                    vlr_list.append(vlr)
                    vlc_list.append(vlc)
                    voa_list.append(vol)

                    iter += 1
                    if iter % 10 == 0:
                        print('iteration:', iter, 'reg loss:', lr, 'cls loss:', lc, 'overall loss:', ol, 'vreg l.:', vlr, 'vcls l.:',vlc,'voa l.:', vol)


        import pickle

        with open('dump.pkl', 'wb') as file:
            pickle.dump([vx_b, vy_b, vrpreds, vlr_list, vlc_list, voa_list, x_b, y_b, rp, cp, tx_, ty_, tw_, th_, px_, py_, pw_, ph_, ttgs, tpx_, tpy_, tpw_, tph_, tarx_, tary_, tarw_, tarh_, train_ground_truth_tensor[f,:,:,:,:], train_selection_tensor[f,:,:,:,:,:], lr_list, lc_list, oa_list, f, l], file)

        ## plot cost development
        ##import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(reg_loss_list, label='reg loss')
        #plt.plot(cls_loss_list, label='cls loss')
        #plt.plot(oal_loss_list, label='overall loss')
        #plt.legend()
        #plt.show()

        # plot image, true boxes and predicted boxes
