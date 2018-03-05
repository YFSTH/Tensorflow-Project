import pdb


def calculateIoU(proposal, ground_truth_box):
    import numpy as np
    
    x1_p = proposal[0]
    y1_p = proposal[1]
    w_p = proposal[2]
    h_p = proposal[3]
    
    gtb_x = ground_truth_box[0]
    gtb_y = ground_truth_box[1]
    gtb_w = ground_truth_box[2]
    gtb_h = ground_truth_box[3]

    # calculate most upper-left (x1,y1) and bottom-right (x2,y2) pixel coordinates of
    # anchor and ground truth box
    
    x2_p = x1_p + w_p - 1
    y2_p = y1_p + h_p - 1
    
    gtb_x, gtb_y, gtb_w, gtb_h = ground_truth_box[0], ground_truth_box[1], ground_truth_box[2], ground_truth_box[3]

    if gtb_w % 2 == 0:
        x1_t = gtb_x - (gtb_w / 2 - 1)
        x2_t = gtb_x + (gtb_w / 2)
        y1_t = gtb_y - (gtb_h / 2 - 1)
        y2_t = gtb_y + (gtb_h / 2)
    else:
        dw = np.floor(gtb_w / 2)
        dh = np.floor(gtb_h / 2)
        x1_t = gtb_x - dw
        x2_t = gtb_x + dw
        y1_t = gtb_y - dh
        y2_t = gtb_y + dh

    if ~(x1_t > x2_p or x1_p > x2_t or y1_t > y2_p or y1_p > y2_t):  # if anchor and ground truth box intersect
        # calculate coordinates of intersection rectangle
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


def selectProposals(iou_threshold, max_n_highest_cls_scores, logits, proposal_tensor, ground_truth_tensor,
                     selection_tensor, training=True):
    # logits will be of shape: (BATCH_SIZE, 16, 16, NUM_ANCHORS*2)
    import numpy as np
    num_collages = len(proposal_tensor)
    if training:
        # only during training time can the proposals be filtered according to their overlap with the mnist images
        proposal_selection_tensor = np.zeros((num_collages, 16, 16, 9, 3))
        # reject proposals which have an IoU < iou_threshold with ground truth box
        for c in range(num_collages):
            for x in range(16):
                for y in range(16):
                    for t in range(9):
                        if selection_tensor[c][0, x, y, t, 0] == 1:
                            prop_x = proposal_tensor[c][0, x, y, t]
                            prop_y = proposal_tensor[c][0, x, y, t + 9]
                            prop_w = proposal_tensor[c][0, x, y, t + 18]
                            prop_h = proposal_tensor[c][0, x, y, t + 27]
                            gtb_x = ground_truth_tensor[c][0, x, y, t]
                            gtb_y = ground_truth_tensor[c][0, x, y, t + 9]
                            gtb_w = ground_truth_tensor[c][0, x, y, t + 18]
                            gtb_h = ground_truth_tensor[c][0, x, y, t + 27]
                            iou = calculateIoU([prop_x, prop_y, prop_w, prop_h], [gtb_x, gtb_y, gtb_w, gtb_h])
                            # save mnist number class in proposal selection tensor
                            proposal_selection_tensor[c, x, y, t, 1] = selection_tensor[c][0, x, y, t, 1]
                            # save iou of proposal with mnist image in selection proposal tensor
                            proposal_selection_tensor[c, x, y, t, 2] = iou

                            if iou >= iou_threshold:
                                # memorize in proposal selection tensor that this proposal is not filtered out
                                proposal_selection_tensor[c, x, y, t, 0] = 1
                            # the proposal tensor will contain: (collage#, x_fm, y_fm, anchor#, [type, mnist_image {only during train phase}, iou {only during train phase}])

    logits = np.array(logits)
    # shape (700, 1, 16, 16, 18)

    # calculate the predicted probability that the proposal catches an object
    probabilities = np.zeros((num_collages, 16, 16, 9)) - 1
    counter, counter2 = 0, 0
    for c in range(num_collages):
        for x in range(16):
            for y in range(16):
                for t in range(9):
                    # perform softmax with max trick
                    x1 = logits[c, 0, x, y, t * 2]
                    x2 = logits[c, 0, x, y, 1 + t * 2]
                    # apply max trick to avoid computational problems
                    x1_ = x1 - max(x1, x2)
                    x2_ = x2 - max(x1, x2)
                    pos_prob = 1 - ( np.exp(x1_) / (np.exp(x1_) + np.exp(x2_)))
                    # save predicted probability in probabilities tensor
                    probabilities[c, x, y, t] = pos_prob

    # select n best proposals using the cls score amongst proposals over all collages
    selection_array = np.squeeze(np.array(selection_tensor))

    # get indices of positive anchors

    idxs_of_pos_anchors = np.where(selection_array[:, :, :, :, 0] == 1)

    # now we have the indices of the probabilities of the positive anchors
    cls_scores_pos_anchs = probabilities[idxs_of_pos_anchors]
    # now we have the cls scores of the positive anchors
    subidxs_of_n_highest_cls_scores = cls_scores_pos_anchs.argsort()[-max_n_highest_cls_scores:]
    choosen_idxs = np.array(idxs_of_pos_anchors)[:, subidxs_of_n_highest_cls_scores]

    # mark the proposal as choosen if it was not sorted out yet
    updated_proposal_sel_tensor = np.zeros((num_collages, 16, 16, 9, 3))
    updated_proposal_sel_tensor[:, :, :, :, 1] = proposal_selection_tensor[:, :, :, :, 1]
    updated_proposal_sel_tensor[:, :, :, :, 2] = proposal_selection_tensor[:, :, :, :, 2]
    iter = 0
    for i in range(max_n_highest_cls_scores):
        choosen_idx = choosen_idxs[:, i]
        idx_in_proposal_sel_tensor = tuple(choosen_idx) + (0,)

        if proposal_selection_tensor[idx_in_proposal_sel_tensor] == 1:
            updated_proposal_sel_tensor[idx_in_proposal_sel_tensor] = 1


    return updated_proposal_sel_tensor




