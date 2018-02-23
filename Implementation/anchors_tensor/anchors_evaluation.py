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
             and the second the number of the optimal ground truth box +++ ground truth tensor
    '''
    import numpy as np
    from anchor_and_ground_truth_box.Anchor import Anchor
    from anchor_and_ground_truth_box.GroundTruthBox import GroundTruthBox
    from anchors_tensor.create_ground_truth_tensor_and_selection_tensor import create_ground_truth_tensor_and_selection_tensor

    num_imgs = len(labels)
    fm_w_index, fm_h_index = batch_anchor_tensor.shape[-2], batch_anchor_tensor.shape[-1]
    num_anchors = batch_anchor_tensor.shape[1] // 4
    anchor_tensor = batch_anchor_tensor[0, :, :, :]
    # is of shape (NUM_ANCHORS*4, feature map width, feature map height)

    ground_truth_tensors = []
    selection_tensors = []
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
                    anchor_objects.append(Anchor(x, y, w, h, w_idx, h_idx, anchor_index))

        # (get labels of specific collage)
        mnist_imgs = labels[collage]
        # for each mnist image in the collage create one ground truth box
        ground_truth_boxes = []
        for mnist_image in enumerate(mnist_imgs):
            if mnist_image[1][0] >= 0: # filter out filler (-9,...,-9) labels
                x = mnist_image[1][1]
                y = mnist_image[1][2]
                w = mnist_image[1][3]
                h = mnist_image[1][4]
                label = mnist_image[1][0]
                ground_truth_boxes.append(GroundTruthBox(x, y, w, h, label))

        # evaluation (anchor <> ground truth box assignment) takes place inside the Anchor object: object-oriented
        # approach is used to determine the appropriate ground truth box the anchor should transform to (if there is
        # any)
        for anchor in anchor_objects:
            for box in ground_truth_boxes:
                anchor.append_ground_truth_box(box)
                for a in anchor_objects:
                    a.evaluate_anchor()


        gtt, slt = create_ground_truth_tensor_and_selection_tensor(anchor_objects, ground_truth_boxes, num_anchors, fm_w_index, fm_h_index)

        # create one ground_truth_tensor with shape (NUM_TENSORS*4, feature map width, feature map height) per collage
        # image
        ground_truth_tensors.append(gtt)

        # create one selection tensor with shape (NUM_TENSORS, feature map width, feature map height, [type, class]) per
        # image, whereas 'type' is the positive xor neutral xor negative evaluation of the tensor and 'class' the mnist
        # number class
        selection_tensors.append(slt)

        #pdb.set_trace()

    return np.array(ground_truth_tensors), np.array(selection_tensors) # should be of shape (NUM_IMAGES, NUM_TENSORS*4, feature map width, feature map height)




