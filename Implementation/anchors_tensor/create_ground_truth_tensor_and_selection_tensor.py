import pdb

def create_ground_truth_tensor_and_selection_tensor(anchor_objects, objects, num_anchors, fm_w_index, fm_h_index):
    '''

    :param anchor_objects:
    :param objects:
    :param num_anchors: number of anchors per feature map position of the last convolutional layer of the used
                        pretrained convolutional net
    :param fm_w_index: feature map width
    :param fm_h_index: feature map height
    :return: ground_truth_tensor and selection_tensor
    '''
    import numpy as np

    assert (len(anchor_objects) == num_anchors * fm_h_index * fm_w_index)

    gtb_tensor = np.zeros((num_anchors*4, fm_w_index, fm_h_index)) - 2
    sel_tensor = np.zeros((num_anchors, fm_w_index, fm_h_index, 2)) - 2

    positive_anchors = 0
    for a in anchor_objects:
        # get box that was assigned to tensor and its coordinates

        box = a.assigned_ground_truth_box


        if box != None:
            positive_anchors += 1
            x = box.x
            y = box.y
            w = box.w
            h = box.h
            # get coordinates of anchors in anchor tensor
            w_idx = a.w_idx
            h_idx = a.h_idx
            anchor_idx = a.anchor_idx
            # create target box in ground truth tensor
            gtb_tensor[anchor_idx, w_idx, h_idx] = x
            gtb_tensor[anchor_idx+num_anchors*1, w_idx, h_idx] = y
            gtb_tensor[anchor_idx+num_anchors*2, w_idx, h_idx] = w
            gtb_tensor[anchor_idx+num_anchors*3, w_idx, h_idx] = h

            # create selection tensor entry
            # get and code anchor type (positive xor negative xor neutral) and mnist class of assigned ground truth box
            tmp = a.type
            if tmp == 'positive':
                anchortype = 1
            elif tmp == 'negative':
                anchortype = 0
            elif tmp == 'neutral':
                anchortype = -1
            mnist_class = a.assigned_ground_truth_box.label
            sel_tensor[anchor_idx, w_idx, h_idx, 0] = anchortype
            sel_tensor[anchor_idx, w_idx, h_idx, 0] = mnist_class

    pdb.set_trace()

    return gtb_tensor, sel_tensor






