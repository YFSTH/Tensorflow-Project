def createProposals(predicted_coordinates, selection_tensor):
    '''
    This method adjusts the coordinates of the region of interest the RPN predicts, thus, 1. it checks whether its
    center coordinate is inside the image frame, else the proposal is marked as negative, and 2. it checks whether the
    whole predicted box lies in the collage frame, else the width and height of the proposal are adjusted to fit. The
    proposals are mapped to the image coordinates (proposal_on_img_tensor) and also to the feature map´s coordinates
    (proposal_on_fm_tensor) for later use
    :param predicted_coordinates: the region of interest coordinates predicted by the box regression of the RPN
    :param selection_tensor: tensor which enables to sort out non-positive anchors proposals and to mark proposals with
                             a center outside the collage frame to be sorted out
    :return: proposal_on_img_tensor: containing the adjusted region proposals with respect to the image coordinates
    :return: proposal_on_fm_tensor: containing the adjusted region proposals with respect to the RPN unput feature
                                    maps´s coordinates
    :return: selection_tensor: sorted out proposals are tagged with a "-9" in the "anchor type" entry, the first entry
                               of the last dimension of the tensor
    '''
    import numpy as np
    # Implementation for single collage at the time
    proposal_on_fm_tensor = np.zeros((1, 16, 16, 36)) - 9
    proposal_on_img_tensor = np.zeros((1, 16, 16, 36)) - 9
    for x in range(16):
        for y in range(16):
            for t in range(9):
                if selection_tensor[0, x, y, t, 0] == 1:
                    # check, whether anchor is positive
                    # get the predicted coordinates of the RPN´s box regression
                    pred_x = np.round(predicted_coordinates[0, x, y, t])
                    pred_y = np.round(predicted_coordinates[0, x, y, t + 9])
                    pred_w = np.round(predicted_coordinates[0, x, y, t + 18])
                    pred_h = np.round(predicted_coordinates[0, x, y, t + 27])
                    # test whether the proposals region exceeds the region given by the image and adjust the shape of
                    # the proposal´s region
                    if pred_x < 0 or pred_y < 0 or pred_x > 256 or pred_y > 256 or pred_w < 0 or pred_h < 0:
                        # if the proposal´s center pixel is outside the image frame deem the proposal to be invalid
                        proposal_on_fm_tensor[0, x, y, t + (np.arange(0, 4) * 9)] = -9
                        proposal_on_img_tensor[0, x, y, t + (np.arange(0, 4) * 9)] = -9
                        # ... and tag it with a "-9" in the "anchor type" entry of the selection tensor
                        selection_tensor[0, x, y, t, 0] = -9
                    else:
                        # proposal (x,y) indicates lower-left edge of proposal
                        prop_x = pred_x - (np.ceil(pred_w / 2) - 1)
                        prop_y = pred_y - (np.ceil(pred_h / 2) - 1)
                        prop_w = pred_w
                        prop_h = pred_h
                        # check how much the coordinates exceed the valid image frame
                        neg_excess_x = -prop_x * (prop_x < 0)
                        neg_excess_y = -prop_y * (prop_y < 0)
                        pos_excess_x = ((prop_x + prop_w) - 256) * ((prop_x + prop_w) > 256)
                        pos_excess_y = ((prop_y + prop_h) - 256) * ((prop_y + prop_h) > 256)
                        if neg_excess_x > 0 and pos_excess_x > 0:
                            # if the proposal exceeds the image borders in both directions in at least one of the
                            # image dimensions then prune the width/height to the full dimension range and place the
                            # upper-left pixel of the proposal inside the image frame
                            prop_x = 0
                            prop_w = 256
                        if neg_excess_y > 0 and pos_excess_y > 0:
                            prop_y = 0
                            prop_h = 256
                        if (neg_excess_x > 0) ^ (pos_excess_x > 0):
                            # If the proposal exceeds at least one border into only one direction then prune the width/
                            # height and if necessary reposition the upper-left proposal pixel
                            if neg_excess_x > 0:
                                prop_x = 0
                                prop_w = prop_w - neg_excess_x
                            if pos_excess_x > 0:
                                prop_w = prop_w - pos_excess_x
                        if (neg_excess_y > 0) ^ (pos_excess_y > 0):
                            if neg_excess_y > 0:
                                prop_y = 0
                                prop_h = prop_h - neg_excess_y
                            if pos_excess_y > 0:
                                prop_h = prop_h - pos_excess_y
                        # assign the proposal´s parameters with respect to the collage frame to the respective
                        # positions of the proposal_on_img_tensor
                        proposal_on_img_tensor[0, x, y, t] = prop_x
                        proposal_on_img_tensor[0, x, y, t + 9] = prop_y
                        proposal_on_img_tensor[0, x, y, t + 18] = prop_w
                        proposal_on_img_tensor[0, x, y, t + 27] = prop_h
                        # assign the proposal´s parameters with respect to the RPNs and Fast R-CNNs input feature
                        # map to the respective entries of the proposal
                        proposal_on_fm_tensor[0, x, y, t] = np.floor(prop_x / 16)
                        proposal_on_fm_tensor[0, x, y, t + 9] = np.floor(prop_y / 16)
                        proposal_on_fm_tensor[0, x, y, t + 18] = np.ceil(prop_w / 16)
                        proposal_on_fm_tensor[0, x, y, t + 27] = np.ceil(prop_h / 16)
    return proposal_on_img_tensor, proposal_on_fm_tensor, selection_tensor