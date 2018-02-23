import pdb

def anchors_evaluation(batch_anchor_tensor, imgs, labels):
    '''
    This function discriminates between positive, negative and neutral anchors and additionally
    assigns for each image to each positive anchor the optimal ground truth box (if such a box exists)
    :param anchors: 3D anchor tensor with the shape (number of anchors * 4, feature map width, feature map height),
                    where the feature map is the final feature map of the pretrained convolutional network that is
                    used for the Faster R-CNN
    :param labels: list of labels, i.e. the training, validation xor test image set, one label contains seven
                   sublabels of the mnist image: 1. class, 2. x, 3. y, 4. w, 5. h, 6. angle, 7. scale
    :return: 5D tensor of shape (number of img, num anchor, feature map width, feature map height, 2), whereas the
             feature map is the feature map of the pretrained convolutional network used for the Faster R-CNN; the
             first entry of the fifth dimension indicates the anchor evaluation (positive=1, neural=0, negative=-1)
             and the second the number of the optimal ground truth box
    '''
    import numpy as np
    from anchor_and_ground_truth_box.Anchor import Anchor
    from anchor_and_ground_truth_box.GroundTruthBox import GroundTruthBox

    num_imgs = len(labels)
    fm_w_index, fm_h_index = batch_anchor_tensor.shape[-2], batch_anchor_tensor.shape[-1]
    num_anchors = batch_anchor_tensor.shape[1] // 4
    anchor_tensor = batch_anchor_tensor[0, :, :, :]
    # is of shape (NUM_ANCHORS*4, feature map width, feature map height)

    # Object-oriented approach is used to determine the the appropriate ground truth box the anchor
    # should transform to (if there is any).

    for collage in range(num_imgs):

        # Create list of anchor objects
        anchor_objects = []
        for w_idx in range(fm_w_index):
            for h_idx in range(fm_h_index):
                for anchor_index in range(num_anchors):
                    x = anchor_tensor[anchor_index, w_idx, h_idx]
                    y = anchor_tensor[anchor_index+num_anchors, w_idx, h_idx]
                    w = anchor_tensor[anchor_index+num_anchors*2, w_idx, h_idx]
                    h = anchor_tensor[anchor_index+num_anchors*3, w_idx, h_idx]
                    anchor_objects.append(Anchor(x, y, w, h, w_idx, h_idx))

        # get labels of specific image
        mnist_imgs = labels[collage]
        # for each mnist image in the collage create one ground truth box
        ground_truth_boxes = []
        for mnist_image in enumerate(mnist_imgs):
            x = mnist_image[1][1]
            y = mnist_image[1][2]
            w = mnist_image[1][3]
            h = mnist_image[1][4]
            label = mnist_image[1][0]
            ground_truth_boxes.append(GroundTruthBox(x, y, w, h, label))

        pdb.set_trace()

        # add ground truth boxes to anchors

        # create one ground_truth_tensor per collage image

    return # ground_truth_tensor_list






    pdb.set_trace()
    #

    output_tensor = np.zeros((num_imgs, num_anchors, ))
