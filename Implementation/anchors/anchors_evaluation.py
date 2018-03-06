import pdb
from anchors.debugging_module import *
from anchors.anchors_selection import *

def anchors_evaluation(batch_anchor_tensor, imgs, labels, load_last_anchors, filename, num_selected, lower_threshold, upper_threshold):
    '''
    This function discriminates between positive, negative and neutral anchors and additionally assigns for each image
    to each positive anchor the optimal ground truth box (if such a box exists)
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
    from anchors.Anchor import Anchor
    from anchors.Ground_truth_box import Ground_truth_box
    from anchors.create_ground_truth_tensor_and_selection_tensor import create_ground_truth_tensor_and_selection_tensor
    import pickle
    import os
    just_reload = False
    if load_last_anchors:
        # if it is indicated that the collages were already created and shall be reused check whether collages already
        # exist if that is the case do nothing
        files = os.listdir('./anchors')
        for f in files:
            if filename+'.pkl' in f:
                just_reload = True
    if just_reload is not True:
        # if the collages shall be newly created
        num_imgs = len(labels)
        # get the shape of the feature map
        fm_w_index, fm_h_index = batch_anchor_tensor.shape[-2], batch_anchor_tensor.shape[-1]
        num_anchors = batch_anchor_tensor.shape[1] // 4
        anchor_tensor = batch_anchor_tensor[0, :, :, :]
        ground_truth_tensors = []
        selection_tensors = []
        for collage in range(num_imgs):
            # for every collage
            # create list of anchor objects
            anchor_objects = []
            for w_idx in range(fm_w_index):
                for h_idx in range(fm_h_index):
                    for anchor_index in range(num_anchors):
                        # get the desired x, y, w, and h parameters
                        x = anchor_tensor[anchor_index, w_idx, h_idx]
                        y = anchor_tensor[anchor_index+num_anchors, w_idx, h_idx]
                        w = anchor_tensor[anchor_index+num_anchors*2, w_idx, h_idx]
                        h = anchor_tensor[anchor_index+num_anchors*3, w_idx, h_idx]
                        # ... and create an anchor with the desired parameters
                        anchor_objects.append(Anchor(x, y, w, h, w_idx, h_idx, anchor_index, lower_threshold, upper_threshold))

            # get mnist images labels of specific collage
            mnist_imgs = labels[collage]
            # for each mnist image in the collage create one ground truth box
            ground_truth_boxes = []
            for mnist_image in enumerate(mnist_imgs):
                if mnist_image[1][0] >= 0: # filter out filler (-9, ..., -9) labels -> to be used as an inputtensor for
                                           # tensorflow later on, all collage label lists must bear the same number of
                                           # mnist image labels, if less then the max number of mnist images was placed
                                           # on the collage (-9, ..., -9) entries are used as fillers
                    x = mnist_image[1][1]
                    y = mnist_image[1][2]
                    w = mnist_image[1][3]
                    h = mnist_image[1][4]
                    label = mnist_image[1][0]
                    # create the ground truth box with the desired x, y, w, and h values
                    ground_truth_boxes.append(Ground_truth_box(x, y, w, h, label))
            # try to add each ground truth box to each anchor (see Anchor.append_ground_truth_box(...) for more
            # informationen)
            for anchor in anchor_objects:
                for box in ground_truth_boxes:
                    anchor.append_ground_truth_box(box)
            # evaluation (anchor <> ground truth box assignment) takes place inside the Anchor object: object-oriented
            # approach is used to determine the appropriate ground truth box the anchor should transform to (if there is
            # any)
            for anchor in anchor_objects:
                anchor.evaluate_anchor()
            # based on the created anchors and ground truth boxes generate a ground truth box and a selection tensor
            # (see create_ground_truth_tensor_and_selection_tensor(...) for more informationen
            gtt, slt = create_ground_truth_tensor_and_selection_tensor(anchor_objects, ground_truth_boxes, num_anchors, fm_w_index, fm_h_index)
            # select a certain amount of anchors (e.g. 256) and try to establish a certain ratio of positive to negative
            # anchors
            slt = anchors_selection(gtt, slt, num_selected)
            # append the ground truth tensor and the selection tensor which was created for the collage to the
            # respective list (in the end there will be one of both tensors for each collage)
            ground_truth_tensors.append(gtt)
            selection_tensors.append(slt)
        # save the created ground truth and selection tensors using the manually specified filename
        with open('anchors/'+filename+'.pkl', 'wb') as file:
            pickle.dump([ground_truth_tensors, selection_tensors], file)
    # if anchors shall just be reloaded
    if just_reload is True:
        with open('anchors/'+filename+'.pkl', 'rb') as file:
            ground_truth_tensors, selection_tensors = pickle.load(file)
    return np.array(ground_truth_tensors), np.array(selection_tensors)




