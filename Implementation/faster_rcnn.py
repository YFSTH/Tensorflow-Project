import os
import numpy as np
import tensorflow as tf

from batch_generator import MNISTCollage


def create_graph():
    """ Creates a graph from saved GraphDef file and returns a saver. """
    with tf.gfile.FastGFile(os.path.join("imagenet", "classify_image_graph_def.pb"), 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')


if __name__ == "__main__":
    # load input data
    mnist = MNISTCollage("datasets")
    
    # load pre-trained ImageNet
    create_graph()

    # read out last pooling layer
    with tf.Session() as sess:
        for image, label in mnist.get_batch(mnist.train_data, mnist.train_labels, 1):
            pool_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            output = sess.run(pool_tensor, feed_dict={'Cast:0': image})
            output = np.squeeze(output)
