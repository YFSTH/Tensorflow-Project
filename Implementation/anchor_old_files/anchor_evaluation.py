import pdb

def anchors_eval(anchors, labels):
    '''
    This function discriminates between positive, negative and neutral anchors and additionally
    assigns for each image to each positive anchor the ground truth box number with the highest overlap
    of intersection.
    :param anchors: 3D anchor tensor with the shape (number of anchors * 4, feature map width, feature map height),
                    where the feature map is the final feature map of the pretrained convolutional network that is
                    used for the Faster R-CNN
    :param labels: list of labels, i.e. the training, validation xor test image set, one label contains seven
                   sublabels of the mnist image: 1. class, 2. x, 3. y, 4. w, 5. h, 6. angle, 7. scale
    :return: 5D tensor of shape (number of img, num anchor, feature map width, feature map height, 2), whereas the feature map
             is the feature map of the pretrained convolutional network used for the Faster R-CNN
    '''
    import numpy as np

    num_imgs = len(labels)
    num_anchors = anchors.shape[1] // 4

    output_tensor = np.zeros((num_imgs, num_anchors, ))
