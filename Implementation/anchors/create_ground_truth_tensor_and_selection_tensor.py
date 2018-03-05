def create_ground_truth_tensor_and_selection_tensor(anchor_objects, ground_truth_boxes, num_anchors, fm_w_index, fm_h_index):
    '''
    The ground truth box tensor will contain the coordinates of the ground truth box which was assigned to the anchor,
    whereas the selection tensor will save whether the anchor is positive, neutral or negative, the mnist number class
    of the assigned ground truth box (if any was assigned) and the IoU of the anchor with the assigned ground truth box
    (or alternatively the max IoU it has with any box if no box was assigned)
    :param anchor_objects: list of anchor objects
    :param ground_truth_boxes: list of ground truth box objects
    :param num_anchors: number of anchors per feature map position of the last convolutional layer which is the input of
                        the RPN
    :param fm_w_index: feature map width
    :param fm_h_index: feature map height
    :return: ground_truth_tensor: shape (1, num_anchor * 4, fm_size, fm_size), where fm_size is the size of the square
                                  shaped feature map, the "4" stands for the for parameters x, y, w and h, whereas the
                                  first 9 layers represent the x coordinates of ground truth boxes assigned to the
                                  anchor (which parameters are saved at the same position in the anchor tensor)
             selection_tensor: shape (1, num_anchors * 4, fm_size, fm_size, 3), whereas the first entry of the fifth
                               dimension indicates the anchors type ("1" for positive, "0" for negative and "-1" for
                               neutral), the second the mnist number class of the assigned ground truth box ("-2" if no
                               box was assigned), and the third entry the IoU with that box
    '''
    import numpy as np

    gtb_tensor = np.zeros((num_anchors*4, fm_w_index, fm_h_index)) - 2
    sel_tensor = np.zeros((num_anchors, fm_w_index, fm_h_index, 3)) - 2
    positive_anchors = 0
    neutral_anchors = 0
    negative_anchors = 0

    for a in anchor_objects:
        # for every anchor
        # get box that was assigned to tensor and its coordinates
        box = a.assigned_ground_truth_box
        # get coordinates of anchor in anchor tensor
        w_idx = a.w_idx
        h_idx = a.h_idx
        anchor_idx = a.anchor_idx
        if box is not None:
            # if there was a box assigned to the anchor
            # get coordinates of assigned box
            x = box.x
            y = box.y
            w = box.w
            h = box.h
            # create entry for target box in ground truth tensor
            gtb_tensor[anchor_idx, w_idx, h_idx] = x
            gtb_tensor[anchor_idx+num_anchors*1, w_idx, h_idx] = y
            gtb_tensor[anchor_idx+num_anchors*2, w_idx, h_idx] = w
            gtb_tensor[anchor_idx+num_anchors*3, w_idx, h_idx] = h
        # get and code anchor type (positive xor negative xor neutral) and mnist class of assigned ground truth box
        t = a.type
        if t == 'positive':
            positive_anchors += 1
            anchortype = 1
        elif t == 'negative':
            negative_anchors += 1
            anchortype = 0
        elif t == 'neutral':
            neutral_anchors += 1
            anchortype = -1
        # get mnist number class of assigned ground truth box
        if box is not None:
            mnist_class = a.assigned_ground_truth_box.label
        else:
            # "-2" is the missing value
            mnist_class = -2
        # create selection tensor entry
        sel_tensor[anchor_idx, w_idx, h_idx, 0] = anchortype
        sel_tensor[anchor_idx, w_idx, h_idx, 1] = mnist_class
        sel_tensor[anchor_idx, w_idx, h_idx, 2] = a.assigned_iou
    # Testing/ debugging purposes:
    print('positive anchors:', positive_anchors)
    print('negative anchors:', negative_anchors)
    print('neutral anchors:', neutral_anchors)
    return gtb_tensor, sel_tensor






