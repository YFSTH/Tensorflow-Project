def create_anchors_tensor(batch_size, num_anchors, original_img_size, feature_map_size, anchors_scales, anchors_aspect_ratios):

    import numpy as np

    X, Y = get_anchors_centers(original_img_size, feature_map_size)

    X = X.reshape((1, feature_map_size, feature_map_size))
    Y = Y.reshape((1, feature_map_size, feature_map_size))
    X = np.repeat(X, num_anchors, axis=0)
    Y = np.repeat(Y, num_anchors, axis=0)
    # should yield tensors of shape: (9, width, height)

    H, W = get_anchors_w_and_h(num_anchors, feature_map_size, anchors_scales, anchors_aspect_ratios)
    # should yield tensors of shape (9, width, height)

    tmp = np.concatenate([X, Y, W, H], axis=0)
    tmp = tmp.reshape((1, num_anchors*4, feature_map_size, feature_map_size))
    anchors_tensor = np.repeat(tmp, batch_size, axis=0)
    # which should be of shape (batch size, num_anchors*4, width, height)

    return anchors_tensor


def get_anchors_centers(original_img_size, feature_map_size):
    '''
    NOTE: All 9 anchors associated to a specific feature map position have the same coordinates
          in the the image space
    :param original_img_size: height or width of the square-shaped image
    :param feature_map_size: height or width of the square-shaped feature map
    :return: matrix containing center coordinates in the image plane for each position of the
             feature map
    '''
    import numpy as np
    # TODO: Mögliche Fehlerquelle
    receptive_field_size = int(original_img_size/feature_map_size)
    X = np.zeros((feature_map_size,feature_map_size))
    Y = np.zeros((feature_map_size,feature_map_size))
    for row in range(feature_map_size):
        for col in range(feature_map_size):
            Y[row,col] = col*receptive_field_size + int(receptive_field_size/2)
            X[row,col] = row*receptive_field_size + int(receptive_field_size/2)
    return X, Y


def get_anchors_w_and_h(num_anchors, feature_map_size, anchors_scales, anchors_aspect_ratios):
    import numpy as np
    H = np.zeros((num_anchors, feature_map_size, feature_map_size))
    W = np.zeros((num_anchors, feature_map_size, feature_map_size))
    for s in enumerate(anchors_scales):
        for r in enumerate(anchors_aspect_ratios):
            A = s[1]**2
            tmp = np.floor((A/r[1])**(1/2))
            H[(s[0]*len(anchors_aspect_ratios))+r[0],:,:] = tmp
            W[(s[0]*len(anchors_aspect_ratios))+r[0],:,:] = np.floor(A/tmp)
    return H, W
