import pdb


def calculate_iou(proposal, ground_truth_box):
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
    
    gtb_x, gtb_y, gtb_w, gtb_h = ground_truth_box.x, ground_truth_box.y, ground_truth_box.w, ground_truth_box.h

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

def selectProposals(iou_threshold, highest_n_cls_scores, logits, proposal_tensor, ground_truth_tensor, selection_tensor):

    # reject proposals which have an IoU < iou_threshold with ground truth box

    # if multiple proposals

    # TODO: muss alles nach allen trainingsepochen geschehen
