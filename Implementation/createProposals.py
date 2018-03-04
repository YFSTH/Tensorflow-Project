import numpy as np
import pickle
import pdb

with open('dump.pkl', 'rb') as file:
    [xt, yt, tpreds, train_selection_tensor, gtt, reg_loss_list, cls_loss_list, oal_loss_list] = pickle.load(file)

collage = xt
label = yt
predicted_coords = tpreds
selection_tensor = train_selection_tensor
# (1, 16, 16, 9, 3)
ground_truth_tensor = gtt
pdb.set_trace()

for x in range(16):
    for y in range(16):
        for t in range(9):
            # check, whether anchor is positive
            if selection_tensor[0, x, y, t, 0] == 1:
                # get predicted coordin
                pred_x = predicted_coords[0, x, y, t]
                pred_y = predicted_coords[0, x, y, t + 9]
                pred_w = predicted_coords[0, x, y, t + 18]
                pred_h = predicted_coords[0, x, y, t + 27]







