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


def selectProposals(iou_threshold, n_highest_cls_scores, logits, proposal_tensor, ground_truth_tensor,
                     selection_tensor, training=True):
    # logits will be of shape: (BATCH_SIZE, 16, 16, NUM_ANCHORS*2)

    import numpy as np

    num_collages = len(proposal_tensor)

    if training:
        # only during training time can the proposals be filtered according to their overlap with the mnist images
        proposal_selection_tensor = np.zeros((num_collages, 16, 16, 9, 3))

        # reject proposals which have an IoU < iou_threshold with ground truth box

        iter = 0
        for c in range(num_collages):
            for x in range(16):
                for y in range(16):
                    for t in range(9):
                        pdb.set_trace()

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
                            iter += 1
                            if iou > iou_threshold:
                                # memorize in proposal selection tensor that this proposal is not filtered out
                                proposal_selection_tensor[c, x, y, t, 0] = 1
                            # the proposal tensor will contain: (collage#, x_fm, y_fm, anchor#, [type, mnist_image {only during train phase}, iou {only during train phase}])

    return proposal_selection_tensor

    # pdb.set_trace()
    # # select n best proposals using the cls score amongst proposals over all collages
    # selection_array = np.array(selection_tensor)
    # logits = np.array(logits)
    #
    #
    # # calculate the predicted probability that the proposal catches an object
    # probabilities = np.zeros((num_collages, 16, 16, 9)) - 1
    # for c in range(num_collages):
    #     for x in range(16):
    #         for y in range(16):
    #             for t in range(9):
    #                 # use softmax to get predicted probability of being an object
    #                 nominator = np.exp(logits[0, x, y, t, 0])
    #                 denominator = np.sum(np.exp(logits[0, x, y, t, 0]) + np.exp(logits[0, x, y, t, 0]))
    #                 # save predicted probability in probabilities tensor
    #                 probabilities[c, x, y, t] = 1 - (nominator / denominator)
    #
    # # get indices of positive anchors
    # idxs_of_pos_anchors = np.where(selection_array[:,0,:,:,:,0]==1)
    # # now we have the indices of the probabilities of the positive anchors
    # cls_scores_pos_anchs = probabilities[idxs_of_pos_anchors]
    # # now we have the cls scores of the positive anchors
    # idxs_of_n_highest_cls_scores = cls_scores_pos_anchs.argsort()[-n_highest_cls_scores]
    # choosen_idxs = np.array(idxs_of_pos_anchors)[idxs_of_n_highest_cls_scores]
    #
    # # mark the proposal as choosen if it was not sorted out yet
    # for idx in choosen_idxs:
    #     if proposal_selection_tensor[idx, 0] == 1:
    #         proposal_selection_tensor[idx, 0] = 3
    #
    # return proposal_selection_tensor
    #
    #
    #
    #
    # # TODO: muss alles nach allen trainingsepochen geschehen
