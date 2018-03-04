import numpy as np
import pickle
import pdb

def createProposals():

    with open('dump.pkl', 'rb') as file:
        [xt, yt, tpreds, train_selection_tensor, gtt, reg_loss_list, cls_loss_list, oal_loss_list] = pickle.load(file)

    collage = xt
    label = yt
    predicted_coords = tpreds
    selection_tensor = train_selection_tensor
    # (1, 16, 16, 9, 3)
    ground_truth_tensor = gtt
    pdb.set_trace()


    # Implementation for single collage at the time
    proposal_on_fm_tensor = np.zeros((1, 16, 16, 36))
    proposal_on_img_tensor = np.zeros((1, 16, 16, 36))
    # shall be of shape (1, FM_x, FM_y, 36)

    for x in range(16):
        for y in range(16):
            for t in range(9):
                # check, whether anchor is positive
                if selection_tensor[0, x, y, t, 0] == 1:

                    # get predicted coordinates
                    pred_x = np.round(predicted_coords[0, x, y, t])
                    pred_y = np.round(predicted_coords[0, x, y, t + 9])
                    pred_w = np.round(predicted_coords[0, x, y, t + 18])
                    pred_h = np.round(predicted_coords[0, x, y, t + 27])

                    # test whether the proposals region exceeds the region given by the image and adjust the shape of
                    # the proposal´s region
                    if pred_x < 0 or pred_y < 0 or pred_x > 256 or pred_y > 256 or pred_w < 0 or pred_h < 0:
                        # if the proposal´s center pixel is outside the image frame deem the proposal to be invalid
                        proposal_on_fm_tensor[1, x, y, t + (np.arange(0, 4) * 9)] = -9
                        proposal_on_img_tensor[1, x, y, t + (np.arange(0, 4) * 9)] = -9
                    else:
                        # proposal (x,y) indicates lower-left edge of proposal
                        prop_x = pred_x - (np.ceil(pred_w / 2) - 1)
                        prop_y = pred_y - (np.ceil(pred_y / 2) - 1)
                        prop_w = np.round(pred_w)
                        prop_h = np.round(pred_h)

                        # check how much the coordinates exceed the valid image frame
                        neg_excess_x = -prop_x * (prop_x < 0)
                        neg_excess_y = -prop_y * (prop_y < 0)
                        pos_excess_x = ((prop_x + prop_w) - 256) * ((prop_x + prop_w) > 256)
                        pos_excess_y = ((prop_y + prop_h) - 256) * ((prop_y + prop_h) > 256)

                        # prune coordinates to valid image region
                        if (neg_excess_x > 0 and pos_excess_x > 0) or (neg_excess_y > 0 and pos_excess_y > 0):
                            # If the proposal exceeds the image borders in both directions in at least one of the
                            # image dimensions then prune the width/height to the full dimension range and place the
                            # upper-left pixel of the proposal inside the image frame
                            if neg_excess_x > 0 and pos_excess_x > 0:
                                pru_x = 0
                                pru_w = 256
                            if neg_excess_y > 0 and pos_excess_y > 0:
                                pru_y = 0
                                pru_h = 256
                        if (neg_excess_x > 0 ^ pos_excess_x > 0) or (neg_excess_y > 0 ^ pos_excess_y > 0):
                            # If the proposal exceeds at least one border into only one direction then prune the width/
                            # height and if necessary reposition the upper-left proposal pixel
                            if neg_excess_x > 0:
                                pru_x = 0
                                pru_w = prop_w - neg_excess_x
                            if pos_excess_x > 0:
                                pru_w = prop_w - pos_excess_x
                            if neg_excess_y > 0:
                                pru_y = 0
                                pru_h = prop_h - neg_excess_y
                            if pos_excess_y > 0:
                                pru_h = prop_h - neg_excess_y

                            # assigne the proposal´s parameters with respect to the collage frame to the respective
                            # positions of the proposal_on_img_tensor
                            proposal_on_img_tensor[0, x, y, t] = pru_x
                            proposal_on_img_tensor[0, x, y, t + 9] = pru_y
                            proposal_on_img_tensor[0, x, y, t + 18] = pru_w
                            proposal_on_img_tensor[0, x, y, t + 27] = pru_h

                            # assign the proposal´s parameters with respect to the RPNs and Fast R-CNNs input feature
                            # map to the respective entries of the proposa
                            proposal_on_fm_tensor[0, x, y, t] = np.ceil(pru_x / 16)
                            proposal_on_fm_tensor[0, x, y, t + 9] = np.ceil(pru_y / 16)
                            proposal_on_fm_tensor[0, x, y, t + 18] = np.ceil(pru_w / 16)
                            proposal_on_fm_tensor[0, x, y, t + 27] = np.ceil(pru_h / 16)

    return proposal_on_img_tensor, proposal_on_fm_tensor




