import numpy as np
from skimage.measure import block_reduce
import tensorflow as tf


def convolutional(input, kernel_shape, stride_size, normalization=False, activation=None):
    """
    Initialize a convolutional layer with given kernel shape and stride size. Normalize batch and use specified
    activation function if desired.

    :param input: tensor with input data
    :param kernel_shape: 4D tensor of [height, width, input channels, output channels]
    :param stride_size: integer with stride size
    :param normalization: flag, if output should normalize or not
    :param activation: activation function
    :return: convoluted feature maps
    """
    fan_in = int(input.shape[1] * input.shape[2])

    if "relu" in str(activation):
        weight_initializer = tf.random_normal_initializer(stddev=2.0/fan_in)
        bias_initializer = tf.constant_initializer(0.1)
    else:
        weight_initializer = tf.random_normal_initializer(stddev=fan_in**(-0.5))
        bias_initializer = tf.constant_initializer(0.0)

    weights = tf.get_variable("weights", kernel_shape, tf.float32, initializer=weight_initializer)
    biases = tf.get_variable("biases", kernel_shape[3], tf.float32, initializer=bias_initializer)

    output = tf.nn.conv2d(input, weights, strides=[1, stride_size, stride_size, 1], padding='SAME') + biases

    if normalization:
        mean, var = tf.nn.moments(output, axes=[0, 1, 2])
        output = tf.nn.batch_normalization(output, mean, var, offset=None, scale=None, variance_epsilon=1e-6)

    if callable(activation):
        output = activation(output)

    return output


def fully_connected(input, n_neurons, normalization=False, activation=None):
    """
    Initialize a fully-connected feed-forward layer with given number of neurons. Normalize batch and use specified
    activation function if desired.

    :param input: tensor with input data
    :param n_neurons: integer with number of neurons
    :param normalization: flag, if output should normalize or not
    :param activation: activation function
    :return: weighted output
    """
    fan_in = int(input.shape[-1])

    if "relu" in str(activation):
        weight_initializer = tf.random_normal_initializer(stddev=2.0 / fan_in)
        bias_initializer = tf.constant_initializer(0.1)
    else:
        weight_initializer = tf.random_normal_initializer(stddev=fan_in ** (-0.5))
        bias_initializer = tf.constant_initializer(0.0)

    weights = tf.get_variable("weights", [fan_in, n_neurons], tf.float32, initializer=weight_initializer)
    biases = tf.get_variable("biases", [n_neurons], tf.float32, initializer=bias_initializer)

    output = tf.matmul(input, weights) + biases

    if normalization:
        mean, var = tf.nn.moments(output, axes=[0])
        output = tf.nn.batch_normalization(output, mean, var, offset=None, scale=None, variance_epsilon=1e-6)

    if callable(activation):
        output = activation(output)

    return output


def roi_pooling(input, proposals, output_shape):
    """
    Initialize a region of interest (ROI) pooling layer with given proposals and output shape. Rescale all region
    propsals to uniform width and height for fully-connected layers.

    :param input: feature maps from a convolutional layer
    :param proposals: 4D tensor for region proposals of [left upper corner x, left upper corner y, width, height]
    :param output_shape: 2D tensor for rescaled output of [width, height]
    :return: uniform region proposals
    """
    roi = input[:, proposals[1]:proposals[1]+proposals[3], proposals[0]:proposals[0]+proposals[2], :]

    if proposals[3] % output_shape[1] != 0:
        roi = np.repeat(roi, output_shape[1], axis=1)
    else:
        proposals[3] = int(proposals[3] / output_shape[1])

    if proposals[2] % output_shape[0] != 0:
        roi = np.repeat(roi, output_shape[0], axis=2)
    else:
        proposals[2] = int(proposals[2] / output_shape[0])

    kernel = (1, proposals[3], proposals[2], 1)
    roi = block_reduce(roi, kernel, np.max)

    return roi


# NOTE: TensorFlow version of the ROI pooling (does not work because of problems with dynamic reshaping)
def tf_roi_pooling(input, proposals, output_shape):
    """
    Initialize a region of interest (ROI) pooling layer with given proposals and output shape. Rescale all region
    propsals to uniform width and height for fully-connected layers.

    :param input: feature maps from a convolutional layer
    :param proposals: 4D tensor for region proposals of [left upper corner x, left upper corner y, width, height]
    :param output_shape: 2D tensor for rescaled output of [width, height]
    :return: uniform region proposals
    """
    roi = input[:, proposals[1]:proposals[1]+proposals[3], proposals[0]:proposals[0]+proposals[2], :]

    if proposals[3] % output_shape[1] != 0:
        tmp = tf.tile(tf.reshape(tf.transpose(roi, [0, 3, 2, 1]), [-1, 1]), [1, output_shape[1]])
        roi = tf.transpose(tf.reshape(tmp, [roi.shape[0], roi.shape[3], roi.shape[2], proposals[3] * output_shape[1]]), [0, 3, 2, 1])
    else:
        proposals[3] = int(proposals[3] / output_shape[1])

    if proposals[2] % output_shape[0] != 0:
        tmp = tf.tile(tf.reshape(tf.transpose(roi, [0, 1, 3, 2]), [-1, 1]), [1, output_shape[0]])
        roi = tf.transpose(tf.reshape(tmp, [roi.shape[0], roi.shape[1], roi.shape[3], proposals[2] * output_shape[0]]), [0, 1, 3, 2])
    else:
        proposals[2] = int(proposals[2] / output_shape[0])

    kernel = [1, proposals[3], proposals[2], 1]
    roi = tf.nn.max_pool(roi, ksize=kernel, strides=kernel, padding='SAME')

    return roi
