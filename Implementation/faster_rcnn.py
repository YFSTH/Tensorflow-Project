import os
import numpy as np
import tensorflow as tf

from batch_generator import MNISTCollage
from vgg16.vgg16 import VGG16


def create_graph():
    """ Creates a graph from saved GraphDef file and returns a saver. """
    with tf.gfile.FastGFile(os.path.join("imagenet", "classify_image_graph_def.pb"), 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')


if __name__ == "__main__":
    # load input data
    mnist = MNISTCollage("datasets")
    
    # load pre-trained VGG16 net
    #create_graph()
    inputs = tf.placeholder(tf.float32, [1, 128, 128, 3])
    graph = VGG16()
    graph.build(inputs)

    # read out last pooling layer
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #train_writer = tf.summary.FileWriter("./summaries/train", tf.get_default_graph())
        for image, label in mnist.get_batch(mnist.train_data, mnist.train_labels, 1):
            result_tensor = sess.graph.get_tensor_by_name('conv5_3/Relu:0')
            output = sess.run(result_tensor, feed_dict={inputs: image})
            print(output.shape)
