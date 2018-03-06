def calculate_iou(proposal, ground_truth_box):
    '''
    Calculates the Intersection of Union of the region of interest predicted by the RPN and the actual ground truth box
    the anchor tried to predict.
    :param proposal: parameter tuple containing x, y, w, and h parameters of the predicted region of interest
    :param ground_truth_box: parameter tuple containing x, y, w, and h parameters of the ground truth box
    :return: Intersection of Union between proposal and ground truth box (=e.g. mist image)
    '''
    import numpy as np
    # extract the proposal´s coordinates
    x1_p = proposal[0]
    y1_p = proposal[1]
    w_p = proposal[2]
    h_p = proposal[3]
    # extract the ground truth box coordinates
    gtb_x = ground_truth_box[0]
    gtb_y = ground_truth_box[1]
    gtb_w = ground_truth_box[2]
    gtb_h = ground_truth_box[3]
    # calculate most upper-left (x1,y1) and bottom-right (x2,y2) pixel coordinates of anchor and ground truth box (
    # remember: the (x, y) values of the proposal already represent the top-left corner, whereas the (x, y) values of
    # the ground truth box represent the center pixel)
    x2_p = x1_p + w_p - 1
    y2_p = y1_p + h_p - 1
    gtb_x, gtb_y, gtb_w, gtb_h = ground_truth_box[0], ground_truth_box[1], ground_truth_box[2], ground_truth_box[3]
    # ... thus the top-left and bottom right corner must first be calculated
    if gtb_w % 2 == 0:
        # if the ground truth box´s size is even
        x1_t = gtb_x - (gtb_w / 2 - 1)
        x2_t = gtb_x + (gtb_w / 2)
        y1_t = gtb_y - (gtb_h / 2 - 1)
        y2_t = gtb_y + (gtb_h / 2)
    else:
        # if the ground truth box´s size is odd
        dw = np.floor(gtb_w / 2)
        dh = np.floor(gtb_h / 2)
        x1_t = gtb_x - dw
        x2_t = gtb_x + dw
        y1_t = gtb_y - dh
        y2_t = gtb_y + dh
    if ~(x1_t > x2_p or x1_p > x2_t or y1_t > y2_p or y1_p > y2_t):
        # if anchor and ground truth box intersect the intersection of union can be calculated using the subsequent
        # algorithm
        # top-left (x1, y1) and bottom-right (x2, y2) coordinates of the intersecting rectangle
        x1_i = max(x1_p, x1_t)
        y1_i = max(y1_p, y1_t)
        x2_i = min(x2_p, x2_t)
        y2_i = min(y2_p, y2_t)
        # areas of anchor and ground truth box and intersection
        area_a = w_p * h_p
        area_t = gtb_w * gtb_h
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        # calculate and return intersection over union
        return intersection / (area_a + area_t - intersection)
    else:
        return 0


def select_proposals(iou_threshold, max_n_highest_cls_scores, logits, proposal_tensor, ground_truth_tensor,
                    selection_tensor, training=True):
    '''
    Select the proposals produced by the RPN based on two criteria: 1. the intersection of union between the proposed
    region and the actual ground truth box must exceed a specified threshold, 2. only max. the n proposals with the
    highest objectness-score (=cls-score) are used. A third possible selection criterion (to select only the top cls-
    scoring proposals for each ground truth box in separate), was not applied, because the number of positive anchors
    remained relatively low.
    :param iou_threshold: only proposals exceeding the threshold are passed on to the ROI pooling layer
    :param max_n_highest_cls_scores: the top n objectness-score proposals are selected
    :param logits: the unprocessed logits, the probabilities of being an object will be extracted by the application
                   of softmax
    :param proposal_tensor: region proposals of the RPN containing the x, y, w, and h parameter of the proposal
    :param ground_truth_tensor: tensor containing the parameters of the actual ground truth box the anchor tried to pre-
                                predict
    :param selection_tensor: tensor containing informationen about the anchor class, the mnist number class of the
                             ground truth box and the IoU between anchor and ground truth box
    :param training: criterion 1, the selection of proposals based on their overlap with the ground truth box they are
                     supposed to predict, can only be applied during training / when the labels are known
    :return: updated_proposal_sel_tensor: updated selection tensor containing information about whether the proposal was
                                          sorted out; structure: (collage#, feature map x position, feature map x pos-
                                                                  ition, anchor#, [proposal type, mnist number class,
                                                                  iou])
    '''
    import numpy as np
    num_collages = len(proposal_tensor)
    if training:
        # only during training time can the proposals be filtered according to their overlap with the mnist images
        proposal_selection_tensor = np.zeros((num_collages, 16, 16, 9, 3))
        for c in range(num_collages):
            for x in range(16):
                for y in range(16):
                    for t in range(9):
                        if selection_tensor[c][0, x, y, t, 0] == 1:
                            # get the x, y, w, and h parameters of the proposals and the ground truth boxes
                            prop_x = proposal_tensor[c][0, x, y, t]
                            prop_y = proposal_tensor[c][0, x, y, t + 9]
                            prop_w = proposal_tensor[c][0, x, y, t + 18]
                            prop_h = proposal_tensor[c][0, x, y, t + 27]
                            gtb_x = ground_truth_tensor[c][0, x, y, t]
                            gtb_y = ground_truth_tensor[c][0, x, y, t + 9]
                            gtb_w = ground_truth_tensor[c][0, x, y, t + 18]
                            gtb_h = ground_truth_tensor[c][0, x, y, t + 27]
                            # calculate the intersection of union between the proposal and the ground truth box
                            iou = calculate_iou([prop_x, prop_y, prop_w, prop_h], [gtb_x, gtb_y, gtb_w, gtb_h])
                            # save mnist number class in proposal selection tensor
                            proposal_selection_tensor[c, x, y, t, 1] = selection_tensor[c][0, x, y, t, 1]
                            # save iou of proposal with mnist image in selection proposal tensor
                            proposal_selection_tensor[c, x, y, t, 2] = iou
                            # reject proposals which have an IoU < iou_threshold with ground truth box
                            if iou >= iou_threshold:
                                # memorize in proposal selection tensor that this proposal is not filtered out
                                proposal_selection_tensor[c, x, y, t, 0] = 1
    logits = np.array(logits)
    # calculate the predicted probability that the proposal catches an object
    probabilities = np.zeros((num_collages, 16, 16, 9)) - 1
    for c in range(num_collages):
        for x in range(16):
            for y in range(16):
                for t in range(9):
                    # perform softmax with max trick to avoid computational problems
                    x1 = logits[c, 0, x, y, t * 2]
                    x2 = logits[c, 0, x, y, 1 + t * 2]
                    x1_ = x1 - max(x1, x2)
                    x2_ = x2 - max(x1, x2)
                    pos_prob = 1 - ( np.exp(x1_) / (np.exp(x1_) + np.exp(x2_)))
                    # save predicted probability in probabilities tensor
                    probabilities[c, x, y, t] = pos_prob
    # select n best proposals using the cls score amongst proposals over all collages
    selection_array = np.squeeze(np.array(selection_tensor))
    # get indices of positive anchors / proposals
    idxs_of_pos_anchors = np.where(selection_array[:, :, :, :, 0] == 1)
    # get cls scores of positive anchors / proposals
    cls_scores_pos_anchs = probabilities[idxs_of_pos_anchors]
    # get the indices of the n highest cls scores (-> note: the indices with respect to the list of indices of the pos-
    # itive anchors)
    subidxs_of_n_highest_cls_scores = cls_scores_pos_anchs.argsort()[-max_n_highest_cls_scores:]
    # get (parts of) the actual indices of the choosen anchors / proposals in the selection tensor
    choosen_idxs = np.array(idxs_of_pos_anchors)[:, subidxs_of_n_highest_cls_scores]
    # copy the mnist number class and iou between proposal and ground truth box
    updated_proposal_sel_tensor = np.zeros((num_collages, 16, 16, 9, 3))
    updated_proposal_sel_tensor[:, :, :, :, 1] = proposal_selection_tensor[:, :, :, :, 1]
    updated_proposal_sel_tensor[:, :, :, :, 2] = proposal_selection_tensor[:, :, :, :, 2]
    # for all selected proposals
    # if there are not enough proposals index range must be pruned
    num_idxs_to_select = min(choosen_idxs.shape[1], max_n_highest_cls_scores)
    for i in range(num_idxs_to_select):
        # build the index
        choosen_idx = choosen_idxs[:, i]
        idx_in_proposal_sel_tensor = tuple(choosen_idx) + (0,)
        # (positively) select the proposal if it was not sorted out yet
        if proposal_selection_tensor[idx_in_proposal_sel_tensor] == 1:
            updated_proposal_sel_tensor[idx_in_proposal_sel_tensor] = 1
    return updated_proposal_sel_tensor




