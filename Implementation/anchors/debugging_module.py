def debugging_module(batch_anchor_tensor, collage, label, gtt, slt, anchor_objects, ground_truth_boxes):
    import numpy as np
    import pdb

    # Note: A single collage image is observed
    import matplotlib.pyplot as plt

    for l in label:
        collage[l[1] - 2:l[1] + 2, l[2] - 2:l[2] + 2, 0] = 125
        collage[l[1] - 2:l[1] + 2, l[2] - 2:l[2] + 2, 1] = 0
        collage[l[1] - 2:l[1] + 2, l[2] - 2:l[2] + 2, 2] = 125
        print('marked label:',label)
        # => labels are correct

    for a in anchor_objects:
        if a.type == 'positive':
            x = int(a.x)
            y = int(a.y)
            print('marked anchor: x:',x,', y:',y,', w: ', a.w,', h:', a.h,', iou:',a.assigned_iou,', w_idx:', a.w_idx,', h_idx:', a.h_idx)
            collage[x - 2:y + 2, y - 2:y + 2, 0] = 255
            collage[x - 2:y + 2, y - 2:y + 2, 1] = 125
            collage[x - 2:y + 2, y - 2:y + 2, 2] = 0
            # => ious seem to be correct
            # => it seems that correctly scaled anchors were choosen


    # todo: test whether gtt and slt are correct


    # TODO: Result: Parameters are correctly assigned but index h_idx, w_idx are wrong for a part of the images



    plt.imshow(collage); plt.show()

    for t in range(1):
        print(gtt[t, :, :])
        print(slt[t, :, :, 0])



    print(label)

    pdb.set_trace()





