### Preparation #######################################################################################################

# Import packages
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from anchors.create_anchors_tensor import create_anchors_tensor
from anchors.anchors_evaluation import anchors_evaluation
from data_generation.batch_generator import MNISTCollage
from data_generation.data_generator import create_collages
from network.layers import convolutional, fully_connected, roi_pooling
from proposals.create_proposals import create_proposals
from proposals.select_proposals import select_proposals
from vgg16.vgg16 import VGG16

# Set class variables

# Image generation
NUM_COLLAGES = 150
COLLAGE_SIZE = 256
MIN_NUM_IMGS = 2
MAX_NUM_IMGS = 4
REPLACEMENT = True
ALLOW_OVERHANG = False
BACKGROUND = 'black'
MIN_SCALING = 3  # original MNIST image size is 28x28
MAX_SCALING = 3
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
ANCHORS_SCALES = [85, 84, 84]
ANCHORS_RATIOS = [1.0, 1.0, 1.0]
NUM_ANCHORS = 9
LOAD_LAST_ANCHORS = True
LOWER_THRESHOLD = 0.30
UPPER_THRESHOLD = 0.60
NUM_SELECTED_ANCHORS = 256

# RPN
REG_TO_CLS_LOSS_RATIO = 10
EPOCHS_TRAINSTEP_1 = 1
RPN_ACTFUN = tf.nn.elu
CKPT_PATH = './checkpoints/'
STORE_RPN = True
RESTORE_RPN = False
RPN_PATH = './checkpoints/rpn.ckpt'
FINALLY_VALIDATE = True

# Fast R-CNN
ROI_FM_SIZE = 8
EPOCHS_TRAINSTEP_2 = 5
LR_FAST = 0.01
STORE_FAST = True
RESTORE_FAST = False
FAST_PATH = './checkpoints/fast.ckpt'

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

# Evaluate anchors and assign the nearest ground truth box to the anchors evaluated as positive and select only a prede-
# fined number of anchors for later use (ideally a ratio of 1:1 positive to negative anchors for the rpn classification
# is achieved)
eval = partial(anchors_evaluation, batch_anchor_tensor=anchors, load_last_anchors=LOAD_LAST_ANCHORS,
               num_selected=NUM_SELECTED_ANCHORS, lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD)
train_ground_truth_tensor, train_selection_tensor = eval(imgs=batcher.train_data, labels=batcher.train_labels, filename='train_anchors')
valid_ground_truth_tensor, valid_selection_tensor = eval(imgs=batcher.valid_data, labels=batcher.valid_labels, filename='valid_anchors')
test_ground_truth_tensor, test_selection_tensor = eval(imgs=batcher.test_data, labels=batcher.test_labels, filename='test_anchors')
# These methods should return two tensors:
# First tensor: Ground truth box tensor of shape (NUM_IMGS, NUM_ANCHORS*4, FM_WIDTH, FM_HEIGHT)
# Second tensor: Selection tensor (NUM_IMGS, NUM_ANCHORS*4, FM_WIDTH, FM_HEIGHT, [ANCHOR_TYPE, MNIST_CLASS, IoU]),
#                where ANCHOR_TYPE is either positive (=1), negative (=0), neutral (=-1) or deactivated
#                (= -3) and MNIST_CLASS indicates the mnist number class of the assigned ground truth mnist image xor
#                '-2' if no ground truth box was assigned

# swap dimensions of anchor tensors to fit the shape of the predicted coordinates tensor of the RPN and add length 1
# zero dimension
swapaxes = lambda x: np.swapaxes(np.swapaxes(x, 1, 2), 2, 3)
anchors = swapaxes(anchors).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4))
train_ground_truth_tensor = swapaxes(train_ground_truth_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4))
train_selection_tensor = swapaxes(train_selection_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3))
valid_ground_truth_tensor = swapaxes(valid_ground_truth_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4))
valid_selection_tensor = swapaxes(valid_selection_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3))
test_ground_truth_tensor = swapaxes(test_ground_truth_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4))
test_selection_tensor = swapaxes(test_selection_tensor).reshape((NUM_COLLAGES, 1, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3))


### Data Flow Graph Construction Phase ################################################################################

### Region Proposal Network RPN

with tf.variable_scope('rpn'):
    # the region proposal network regresses the parameters of each positive anchor onto the parameters of the assigned
    # ground truth box (=mnist image) and uses anchors evaluated as positive and anchors evaluated as negative to learn
    # to predict the likelihood that the region encompassed by the anchors region contains an object (objectness / cls
    # score)

    with tf.name_scope('placeholders'):
        # Placeholders for input collage(s batch) and associated label list (one label for each small mnist image added
        # to the collage)
        X = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, VGG_FM_NUM])
        Y = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_NUM_IMGS, 7])
        # Placeholders for the anchors coordinates, the groundtruth coordinates, thus the coordinates of the ground
        # truth box the respective anchor was assigned to (if any) and the selection tensor harboring e.g. information
        # about the type of the anchor (positive / neutral / negative)
        anchor_coordinates = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4])
        groundtruth_coordinates = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS*4])
        selection_tensor = tf.placeholder(tf.float32, [BATCH_SIZE, VGG_FM_SIZE, VGG_FM_SIZE, NUM_ANCHORS, 3])

    with tf.variable_scope('pre_heads_layer'):
        # first convolutional layer of the rpn, resembling the shifting of the network over the input feature map
        prehead_conv = convolutional(X, [3, 3, 512, 512], 1, False)

    with tf.name_scope('regression_head'):
        # in the regression head the bounding box regression will be performed
        with tf.variable_scope('predictions'):
            # the regression weights are resembled by the weights of the convolutional layer (the intercept by the bias
            # of the convolutional layer) the output of the convolutional layer resembles the predicted coordinates
            predicted_coordinates = convolutional(prehead_conv, [1, 1, 512, 36], 1, False)

        with tf.variable_scope('regression_parametrization'):
            # perform boxregression parametrization by transforming the predicted x and y coordinates and width and
            # height (= the proposal coordinates) to yield the transformed predicted and true (=target) coordinate para-
            # meters used for the regression loss
            tx_xor_ty = lambda t1, t2, t3: tf.divide(tf.subtract(t1, t2), t3)
            tw_xor_th = lambda t1, t2: tf.log(tf.divide(t1, t2))
            # the following operations yield the transformed predicted parameters
            tx = tx_xor_ty(predicted_coordinates[:, :, :, 0:9], anchor_coordinates[:, :, :, 0:9],
                           anchor_coordinates[:, :, :, 18:27])
            ty = tx_xor_ty(predicted_coordinates[:, :, :, 9:18], anchor_coordinates[:, :, :, 9:18],
                           anchor_coordinates[:, :, :, 27:36])
            tw = tw_xor_th(predicted_coordinates[:, :, :, 18:27], anchor_coordinates[:, :, :, 18:27])
            th = tw_xor_th(predicted_coordinates[:, :, :, 27:36], anchor_coordinates[:, :, :, 27:36])
            t_predicted = tf.concat([tx, ty, tw, th], axis=0)
            # remove nans caused by division
            t_predicted = tf.where(tf.is_nan(t_predicted), tf.zeros_like(t_predicted), t_predicted)
            # the following operations yield the transformed target / ground truth box parameters
            ttx = tx_xor_ty(groundtruth_coordinates[:, :, :, 0:9], anchor_coordinates[:, :, :, 0:9],
                            anchor_coordinates[:, :, :, 18:27])
            tty = tx_xor_ty(groundtruth_coordinates[:, :, :, 9:18], anchor_coordinates[:, :, :, 9:18],
                            anchor_coordinates[:, :, :, 27:36])
            ttw = tw_xor_th(groundtruth_coordinates[:, :, :, 18:27], anchor_coordinates[:, :, :, 18:27])
            tth = tw_xor_th(groundtruth_coordinates[:, :, :, 27:36], anchor_coordinates[:, :, :, 27:36])
            t_target = tf.concat([ttx, tty, ttw, tth], axis=0)
            # remove nans caused by division
            t_target = tf.where(tf.is_nan(t_target), tf.zeros_like(t_target), t_target)

            with tf.variable_scope('regression_loss'):
                def smooth_l1_loss(raw_deviations, selection_tensor):
                    # select deviations for anchors marked as positive
                    activation_value = tf.constant(1.0, dtype=tf.float32)
                    filter_plane = tf.cast(tf.equal(selection_tensor[:, :, :, :, 0], activation_value), tf.float32)
                    filter_tensor = tf.tile(filter_plane, [4, 1, 1, 1])
                    filtered_tensor = tf.multiply(raw_deviations, filter_tensor)
                    filtered_tensor = tf.where(tf.is_nan(filtered_tensor), tf.zeros_like(filtered_tensor),
                                               filtered_tensor)
                    # calculate the smooth l1 loss
                    absolute_deviations = tf.abs(filtered_tensor)
                    # case 1: l(x), |x| < 1
                    case1_sel_tensor = tf.less(absolute_deviations, 1)
                    case1_deviations = tf.multiply(absolute_deviations, tf.cast(case1_sel_tensor, tf.float32))
                    case1_output = tf.multiply(tf.square(case1_deviations), 0.5)
                    # case 2: otherwise
                    case2_sel_tensor = tf.greater_equal(absolute_deviations, 1)
                    case2_output = tf.subtract(absolute_deviations, 0.5)
                    case2_output = tf.multiply(case2_output, tf.cast(case2_sel_tensor, tf.float32))
                    smooth_anchor_losses = case1_output + case2_output
                    # sum smooth anchor loss over all anchors
                    unnormalized_reg_loss = tf.reduce_sum(smooth_anchor_losses)
                    # ... and normalize it by the total number of anchors
                    normalized_reg_loss = tf.truediv(unnormalized_reg_loss, tf.cast((VGG_FM_SIZE ** 2) * 9, tf.float32))
                    return normalized_reg_loss
                # calculate the raw deviations between the transformed prediction and target parameters
                raw_deviations = tf.subtract(t_predicted, t_target)
                # ... and calculate the smooth l1 loss
                rpn_reg_loss_normalized = smooth_l1_loss(raw_deviations, selection_tensor)

        with tf.variable_scope('classification_head'):
            # predicts the probability that the region encompassed by the respective anchor contains an object (object-
            # ness score)
            # the following convolutional layer produces two logits for each anchor, one logit represents the logit for
            # not being an object, the other for being an object
            clshead_conv1 = convolutional(prehead_conv, [1, 1, 512, NUM_ANCHORS * 2], 1, False, RPN_ACTFUN)

            with tf.variable_scope('classification_loss'):
                logits = tf.reshape(clshead_conv1, [BATCH_SIZE * VGG_FM_SIZE * VGG_FM_SIZE * NUM_ANCHORS, 2])
                # get the anchor type ("1" for positive anchor, "0" for negative, "-1" for neutral and "-3" for de-
                # activated) out of the selection tensor
                reshaped_targets = tf.reshape(selection_tensor[:, :, :, :, 0],
                                              [BATCH_SIZE * VGG_FM_SIZE * VGG_FM_SIZE * NUM_ANCHORS, 1])
                # include only positive and negative anchors (the amount is limit by the anchor selection procedure per-
                # formed by anchor_selection(...))
                inclusion_idxs = tf.greater_equal(reshaped_targets, 0)
                # filter out all neutral anchors
                tmp2 = tf.boolean_mask(tf.reshape(logits[:, 0], [tf.shape(logits)[0], 1]), inclusion_idxs)
                tmp3 = tf.boolean_mask(tf.reshape(logits[:, 1], [tf.shape(logits)[0], 1]), inclusion_idxs)
                # also the logits of the not-to-be-used anchors have to be filtered out
                logits_filtered = tf.concat(
                    [tf.reshape(tmp2, [tf.shape(tmp2)[0], 1]), tf.reshape(tmp3, [tf.shape(tmp3)[0], 1])], axis=1)
                # create target vector
                # filter label entries according to the filtered logits
                sampled_targets = tf.reshape(tf.boolean_mask(reshaped_targets, inclusion_idxs), [tf.shape(tmp3)[0], 1])
                ones = tf.ones_like(sampled_targets)
                # get indices of positive anchors
                idx = tf.equal(sampled_targets, 1)
                # get indices of positive anchors
                idxi = tf.not_equal(sampled_targets, 1)
                # create labels
                targets_filtered = tf.concat(
                    [tf.multiply(ones, tf.cast(idxi, tf.float32)), tf.multiply(ones, tf.cast(idx, tf.float32))], axis=1)
                # calculate the accuracy
                predictions = tf.round(tf.nn.softmax(logits_filtered, axis=1))
                accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, targets_filtered), tf.float32))
                # calculate the cross entropy loss
                rpn_cls_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=targets_filtered,
                                                                                     logits=logits_filtered))
                # and normalize it by the number of selected anchors
                rpn_cls_loss_normalized = tf.truediv(rpn_cls_loss, tf.cast(NUM_SELECTED_ANCHORS, tf.float32))

        with tf.name_scope('overall_loss'):
            # calculation of the overall loss of the rpn
            overall_loss = rpn_cls_loss_normalized + REG_TO_CLS_LOSS_RATIO * rpn_reg_loss_normalized

        with tf.variable_scope('costs_and_optimization'):
            # optimization by adam optimizer with piecewise decreasing learning rate
            global_step = tf.Variable(0, trainable=False)
            boundaries = [1200, 1600]
            values = [0.001, 0.0001, 0.000005]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            rpn_train_op = tf.train.AdamOptimizer(learning_rate).minimize(overall_loss, global_step=global_step)

### Fast R-CNN

with tf.variable_scope('fast_rcnn'):
    rois = tf.placeholder(tf.float32, [BATCH_SIZE, ROI_FM_SIZE, ROI_FM_SIZE, VGG_FM_NUM/2])
    classes = tf.placeholder(tf.int64, [BATCH_SIZE])
    boxes = tf.placeholder(tf.float32, [BATCH_SIZE, 4])

    with tf.variable_scope('layer_6'):
        fc6 = fully_connected(tf.reshape(rois, [-1, np.prod(rois.shape[1:])]), 1024, False, tf.nn.leaky_relu)

    #with tf.variable_scope('layer_7'):
    #    fc7 = fully_connected(fc6, 1024, False, tf.nn.leaky_relu)

    with tf.variable_scope('bbox_pred'):
        bbox_pred = fully_connected(fc6, 4, False, tf.nn.leaky_relu)
        bbox_diff = bbox_pred - boxes
        bbox_case_1 = 0.5 * tf.pow(bbox_diff, 2) * tf.cast(tf.less(tf.abs(bbox_diff), 1), tf.float32)
        bbox_case_2 = (tf.abs(bbox_diff) - 0.5) * tf.cast(tf.greater_equal(tf.abs(bbox_diff), 1), tf.float32)
        bbox_loss = tf.reduce_mean(tf.reduce_sum(bbox_case_1 + bbox_case_2, axis=1) / (VGG_FM_SIZE**2 * NUM_ANCHORS))

    with tf.variable_scope('cls_score'):
        cls_score = fully_connected(fc6, 10, False, tf.nn.leaky_relu)
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classes, logits=cls_score))
        cls_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(cls_score), 1), classes), tf.float32))


    fast_loss = cls_loss + bbox_loss
    fast_train = tf.train.AdamOptimizer(LR_FAST).minimize(fast_loss)


### Model initialization nodes

with tf.name_scope('model_initializers'):
    rpn_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rpn'))
    fast_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fast_rcnn'))


### Model saving nodes

with tf.name_scope('model_savers'):
    rpn_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rpn'))
    fast_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fast_rcnn'))


### Execution Phase ###################################################################################################

if __name__ == "__main__":

    # Load pre-trained VGG16 and get handle on input placeholder
    images = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    vgg16 = VGG16()
    vgg16.build(images)

    # start the tensorflow session
    with tf.Session() as sess:
        # Initialize xor restore the required sub-models
        restore_xor_init = lambda restore, saver, path, ini: saver.restore(sess, path) if restore else sess.run(ini)
        if RESTORE_RPN:
            restore_xor_init(RESTORE_RPN, rpn_saver, RPN_PATH, rpn_init)
        if RESTORE_FAST:
            restore_xor_init(RESTORE_FAST, fast_saver, FAST_PATH, fast_init)
        else:
            sess.run(tf.global_variables_initializer())

        #train_writer = tf.summary.FileWriter("./summaries/train", tf.get_default_graph())

        # define variables for saving the diverse outputs of the rpn and fast r-cnn and variables used for later usage
        logits_ = []
        train_proposals_img = []
        train_proposals_fm = []
        valid_proposals_img = []
        valid_proposals_fm = []
        feature_maps = []
        train_step = 0
        collage_copy = None
        labels_copy = None
        proposed_coordinates = None
        selection_tensor_copy = []
        reg_loss_list = []
        cls_loss_list = []
        oal_loss_list = []
        accuracy_list = []

        # rpn training epochs (training step 1)
        for epoch in range(EPOCHS_TRAINSTEP_1):
            # get collage and associated label list
            for X_batch, Y_batch, first, last in batcher.get_batch(BATCH_SIZE):
                # feed the input collage to the pretrained vgg and obtain feature maps tensor used as input for the rpn
                result_tensor = sess.graph.get_tensor_by_name('conv5_3/Relu:0')
                vgg16_conv5_3_relu = sess.run(result_tensor, feed_dict={images: X_batch})
                # feed feature maps tensor to rpn and extract accuracy, losses, and predictions and logits used for
                # later processing
                _, accu, rp, logits__, lr, lc, ol = sess.run(
                    [rpn_train_op, accuracy, predicted_coordinates, clshead_conv1, rpn_reg_loss_normalized,
                     rpn_cls_loss_normalized, overall_loss],
                    feed_dict={X: vgg16_conv5_3_relu,
                               Y: Y_batch,
                               anchor_coordinates: anchors[first],
                               groundtruth_coordinates: train_ground_truth_tensor[first],
                               selection_tensor: train_selection_tensor[first]}
                )

                collage_copy = X_batch
                labels_copy = Y_batch
                proposed_coordinates = rp
                reg_loss_list.append(lr)
                cls_loss_list.append(lc)
                oal_loss_list.append(ol)
                accuracy_list.append(accu)
                selection_tensor_copy = train_selection_tensor[first]

                if epoch + 1 == EPOCHS_TRAINSTEP_1:
                    proposal_img, proposal_fm, train_selection_tensor[first] = create_proposals(proposed_coordinates,
                                                                                                selection_tensor_copy)
                    train_proposals_img.append(proposal_img)
                    train_proposals_fm.append(proposal_fm)
                    feature_maps.append(vgg16_conv5_3_relu)

                logits_.append(logits__)

                if train_step % 10 == 0:
                    print('iteration:', train_step, 'reg loss:', lr, 'cls loss:', lc, 'overall loss:', ol, 'accuracy:',
                          np.round(accu, 2))
                train_step += 1


        # plot the losses development and the arruracy
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(reg_loss_list, color='red', alpha=0.75, label='regression loss')
        plt.plot(cls_loss_list, color='blue', alpha=0.75, label='classification loss')
        plt.plot(oal_loss_list, color='gray', alpha=0.75, label='overall loss')
        plt.legend(fontsize=14)
        plt.title('development of the rpn´s losses', fontweight='bold', fontsize=15)
        plt.xlabel('iteration #', fontsize=15)
        plt.ylabel('loss [different measures!]', fontsize=15)
        plt.subplot(1, 2, 2)
        plt.plot(accuracy_list, color='blue', alpha=0.75)
        plt.legend(fontsize=14)
        plt.title('accuracy of objectness classification', fontweight='bold', fontsize=15)
        plt.xlabel('iteration #', fontsize=15)
        plt.ylabel('accuracy', fontsize=15)
        plt.show()

        # select proposals according to IoU with mnist image and the cls score
        print('select proposals')
        proposal_selection_tensor = select_proposals(iou_threshold=0.15, max_n_highest_cls_scores=9999, logits=logits_,
                                                    proposal_tensor=train_proposals_img,
                                                    ground_truth_tensor=train_ground_truth_tensor,
                                                    selection_tensor=train_selection_tensor, training=True)

        # start the training of the Fast R-CNN
        print('training step 2 started')

        fast_loss_history = []
        fast_accu_history = []

        for epoch in range(EPOCHS_TRAINSTEP_2):
            for n, image in enumerate(feature_maps):
                for i, j, k in np.ndindex(16, 16, 9):
                    if train_selection_tensor[n][:, i, j, k, 0] == 1:
                        # create bounding box from proposals
                        bounding_box = np.zeros(4, dtype=np.int32)
                        bounding_box[0] = train_proposals_fm[n][:, i, j, k]
                        bounding_box[1] = train_proposals_fm[n][:, i, j, 9+k]
                        bounding_box[2] = train_proposals_fm[n][:, i, j, 18+k]
                        bounding_box[3] = train_proposals_fm[n][:, i, j, 27+k]

                        # bring regions of interest (ROI) to same size
                        pool5 = roi_pooling(image[:, :, :, 256:512], bounding_box, [ROI_FM_SIZE, ROI_FM_SIZE])

                        # create ground truth bounding box from proposals
                        gt_bounding_box = np.zeros((BATCH_SIZE, 4))
                        gt_bounding_box[:, 0] = train_proposals_img[n][:, i, j, k]
                        gt_bounding_box[:, 1] = train_proposals_img[n][:, i, j, 9 + k]
                        gt_bounding_box[:, 2] = train_proposals_img[n][:, i, j, 18 + k]
                        gt_bounding_box[:, 3] = train_proposals_img[n][:, i, j, 27 + k]

                        # get MNIST class
                        gt_class = train_selection_tensor[n][:, i, j, k, 1]

                        _, f_loss, c_loss, r_loss, accu = sess.run(
                            [fast_train, fast_loss, cls_loss, bbox_loss, cls_accuracy],
                            feed_dict={rois: pool5, classes: gt_class, boxes: gt_bounding_box}
                        )

                        fast_loss_history.append([f_loss, c_loss, r_loss])
                        fast_accu_history.append(accu)

                print("Processed images in epoch " + str(epoch) + ": " + str(n))

        #storer = lambda boolean, saver, filename: saver.save(sess, CKPT_PATH + filename) if boolean else None
        #storer(STORE_RPN, rpn_saver, 'rpn.ckpt')
        #storer(STORE_FAST, fast_saver, 'fast.ckpt')

