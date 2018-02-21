

### Preparation #######################################################################################################


# Import packages
from anchor_tensor_generation.create_anchor_tensors import *
import os
import numpy as np
import tensorflow as tf
from network.layers import convolutional, fully_connected
from batch_generator import MNISTCollage
from vgg16.vgg16 import VGG16
import pdb
from data_generation.data_gen import *

# Set class variables
NUM_COLLAGES = 5
COLLAGE_SIZE = 128
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
BATCH_SIZE = 1
IMG_SIZE = 128
FM_SIZE = 8
ANCHORS_SCALES = [64, 32, 16]
ANCHORS_RATIOS = [2, 1, 0.5]
NUM_ANCHORS = 9

# Generate images xor load them if they already exist
create_collages(num_collages=NUM_COLLAGES, collage_size=COLLAGE_SIZE, min_num_imgs=MIN_NUM_IMGS,
                max_num_imgs=MAX_NUM_IMGS, replacement=REPLACEMENT, allow_overhang=ALLOW_OVERHANG,
                background=BACKGROUND, min_scaling=MIN_SCALING, max_scaling=MAX_SCALING, scaling_steps=SCALING_STEPS,
                counterclock_angle=COUNTERCLOCK_ANGLE, clockwise_angle=CLOCKWISE_ANGLE, rotation_steps=ROTATION_STEPS)

# Import input batch generator
Batcher = MNISTCollage('./data_generation')

# For debugging: Examine image and labels of batches
#for x,y in Batcher.get_batch(4):
#    pdb.set_trace()

# Import anchor tensors
anchors_tensor = create_anchor_tensor(BATCH_SIZE, NUM_ANCHORS, IMG_SIZE, FM_SIZE, ANCHORS_SCALES, ANCHORS_RATIOS)
# shape: (batchsize, num_anchors*4, feature map height, feature map width)


### Data Flow Graph Construction Phase ################################################################################


#def create_graph():
#   """ Creates a graph from saved GraphDef file and returns a saver. """
#   with tf.gfile.FastGFile(os.path.join("imagenet", "classify_image_graph_def.pb"), 'rb') as file:
#       graph_def = tf.GraphDef()
#       graph_def.ParseFromString(file.read())
#       tf.import_graph_def(graph_def, name='')


### Region Proposal Network RPN

with tf.name_scope('RPN'):
    X = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3], name='input_placeholder')
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_NUM_IMGS, 7])
    # TODO: Might be sufficient to just hand over the classes of the single mnist images
    values_anchors = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_ANCHORS*4, FM_SIZE, FM_SIZE])
    values_truth = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_ANCHORS*4, FM_SIZE, FM_SIZE])

    with tf.name_scope('pre_heads_layers'):
        prehead_conv = convolutional(X, [3, 3, 512, 512], 1, True, tf.nn.relu)

    with tf.name_scope('regression_head'):
        values_predicted = convolutional(prehead_conv, [1, 1, 512, 32], 1, True)
        # have (N,NUM_ANCHORS*4,W,H)


        #conv1_transposed = tf.transpose(conv1, [0,2,3,1])
        #N, W, H, K = tf.shape(conv1_transposed)
        #conv1_reshaped = tf.reshape(conv1_transposed, [int((N*W*H*K)/4), 4])
        #prediction = 0

    with tf.name_scope('classification_head'):
        clshead_conv1 = convolutional(prehead_conv, [1, 1, 512, 32], 1, True, tf.nn.relu)

    with tf.name_scope('costs_and_optimization'):
        pass



### Execution Phase ###################################################################################################


if __name__ == "__main__":

    # load input data
    mnist = MNISTCollage("datasets")
    
    # load pre-trained VGG16 net
    #create_graph()
    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    graph = VGG16()
    graph.build(inputs)

    # read out last pooling layer
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #train_writer = tf.summary.FileWriter("./summaries/train", tf.get_default_graph())

        for X_batch, Y_batch in Batcher.get_batch(BATCH_SIZE):
            # X_batch will be list

            # TODO: Create ground truth boxregression tensor (here, after batching)
            # TODO: Create predcited boxregression tensor (inside the graph?)

            sess.run([prehead_conv], feed_dict={X: X_batch,
                                                Y: Y_batch,
                                                values_anchors: anchors_tensor,
                                                })

            pass

        #for image, label in mnist.get_batch(mnist.train_data, mnist.train_labels, 1):
        #    result_tensor = sess.graph.get_tensor_by_name('conv5_3/Relu:0')
        #    vgg16_conv5_3_relu = sess.run(result_tensor, feed_dict={inputs: image})
        #    print(vgg16_conv5_3_relu.shape)
