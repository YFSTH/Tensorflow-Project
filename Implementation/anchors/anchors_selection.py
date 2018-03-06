def anchors_selection(ground_truth_tensor, selection_tensor, num_selected):
    '''
    Not all of the anchors are used for the regression and classification task. For the regression only positive anchors
    are used as training cases and for the classification a specified number of anchors (e.g. 256) is used where the
    ratio of positive to negative anchors is maximally 1
    :param ground_truth_tensor:
    :param selection_tensor:
    :param num_selected:
    :return: updated selection tensor, by setting the first entry of the last dimension to -3 it is indicated that the
             achor won´t be used
    '''
    import numpy as np
    # get number of anchors per feature map position and size of feature map
    num_anchors = selection_tensor.shape[0]
    fm_w = selection_tensor.shape[1]
    fm_h = selection_tensor.shape[2]
    # select the coordinates of the x parameter of anchors which are either positive xor negative
    valid_negative_anchor_coordinates = []
    valid_positive_anchor_coordinates = []
    for a in range(num_anchors):
        for w in range(fm_w):
            for h in range(fm_h):
                if selection_tensor[a, w, h, 0] == 1:
                    valid_positive_anchor_coordinates.append([a, w, h])
                elif selection_tensor[a, w, h, 0] == 0:
                    valid_negative_anchor_coordinates.append([a, w, h])
                else:
                    # set neutral anchors to 'not selected' (= -3)
                    selection_tensor[a, w, h, 0] = -3
    # determine how many anchors where assigned to a ground truth box
    num_valid_anchors = len(valid_negative_anchor_coordinates) + len(valid_positive_anchor_coordinates)
    # enshure right ratio of positive and negative anchors
    # measure the excess of negative vs positive anchors with respect to the ideal ratio of 1:1 positive-to-negative
    # anchors
    excess_negative_anchors = len(valid_negative_anchor_coordinates) - np.floor(num_selected / 2)
    excess_positive_anchors = len(valid_positive_anchor_coordinates) - np.floor(num_selected / 2)
    # Case 1: There are more of both anchor classes than required
    if excess_positive_anchors >= 0 and excess_negative_anchors >= 0:
        to_be_deactivated_positive_anchors = excess_positive_anchors
        to_be_deactivated_negative_anchors = excess_negative_anchors
    # Case 2: There are not enough positive anchors but enough negative to compensate
    elif excess_positive_anchors <= 0 and excess_negative_anchors >= abs(excess_positive_anchors):
        to_be_deactivated_positive_anchors = 0
        to_be_deactivated_negative_anchors = len(valid_negative_anchor_coordinates) - np.floor(num_selected / 2) - abs(excess_positive_anchors)
    #elif excess_negative_anchors <= 0 and excess_positive_anchors >= abs(excess_negative_anchors):
    #    to_be_deactivated_negative_anchors = 0
    #    to_be_deactivated_positive_anchors = len(valid_positive_anchor_coordinates) - np.floor(num_selected / 2) - abs(excess_negative_anchors)
    else:
        # not enough of both types of anchors
        to_be_deactivated_negative_anchors = 0
        to_be_deactivated_positive_anchors = 0

    if to_be_deactivated_positive_anchors > 0:
        # choose the anchors to un-select and unselect them by assigning -3 to anchor type param
        # located in the selection tensor, to indicate that the anchor won´t be used
        choosen_idxs = np.random.choice(np.arange(0,len(valid_positive_anchor_coordinates)),
                                        int(to_be_deactivated_positive_anchors), replace=False)
        for excess_anchor in choosen_idxs:
            a, x, y = valid_positive_anchor_coordinates[excess_anchor]
            # get the position of the x-center of the anchor
            selection_tensor[a, x, y, 0] = -3

    if to_be_deactivated_negative_anchors > 0:
        # choose the anchors to un-select and unselect them by assigning -3 to anchor type param
        # located in the selection tensor, to indicate that the anchor won´t be used
        choosen_idxs = np.random.choice(np.arange(0, len(valid_negative_anchor_coordinates)),
                                        int(to_be_deactivated_negative_anchors), replace=False)
        for excess_anchor in choosen_idxs:
            # get the position of the x-center of the anchor
            a, x, y = valid_negative_anchor_coordinates[excess_anchor]
            # and and mark the anchor as deactivated in the respective entry of the selection tensor
            selection_tensor[a, x, y, 0] = -3
    return selection_tensor

