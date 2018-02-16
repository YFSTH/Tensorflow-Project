import os
import numpy as np
import tensorflow as tf


def create_graph():
    """ Creates a graph from saved GraphDef file and returns a saver. """
    with tf.gfile.FastGFile(os.path.join("imagenet", "classify_image_graph_def.pb"), 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        _ = tf.import_graph_def(graph_def, name='')


if __name__ == "__main__":
    # load test image
    image = tf.gfile.FastGFile(os.path.join("imagenet", "cropped_panda.jpg"), 'rb').read()
    
    # load DFG
    create_graph()
    train_writer = tf.summary.FileWriter("./summaries/train", tf.get_default_graph())

    # read out last pooling layer
    with tf.Session() as sess:
        pool_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        output = sess.run(pool_tensor, feed_dict={'DecodeJpeg/contents:0': image})
        output = np.squeeze(output)
