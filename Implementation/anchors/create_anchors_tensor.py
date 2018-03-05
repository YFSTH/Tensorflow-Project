def create_anchors_tensor(batch_size, num_anchors, original_img_size, feature_map_size, anchors_scales, anchors_aspect_ratios):
    '''
    Creates an multidimensional matrix for representing the coordinates of all generated anchors for the collage(s).
    :param batch_size: number of images fed to the network per batch (in our case 1)
    :param num_anchors: number of anchors per position in the input feature map of the RPN
    :param original_img_size: size of the collage (square collages are assumed)
    :param feature_map_size: size of the feature map (square feature maps are assumed)
    :param anchors_scales: scales of the desired anchors
    :param anchors_aspect_ratios: aspect ratios of the desired anchors
    :return: returns anchor tensor of shape (1, 16, 16, 9*4) when the feature map is of size 16x16 and there are 9
             anchors per position, the "4" stands for the x, y, w and h parameter of the anchor, where e.g. the first
             9 layers save the x-coordinates of the anchors.
    '''

    import numpy as np
    # create center coordinates of all the anchors of the collage
    X, Y = get_anchors_centers(original_img_size, feature_map_size)
    # ... and reshape and repeat them
    X = X.reshape((1, feature_map_size, feature_map_size))
    Y = Y.reshape((1, feature_map_size, feature_map_size))
    X = np.repeat(X, num_anchors, axis=0)
    Y = np.repeat(Y, num_anchors, axis=0)
    # create width and heigt of all anchors of the collage
    H, W = get_anchors_w_and_h(num_anchors, feature_map_size, anchors_scales, anchors_aspect_ratios)
    # concatenate the anchors parameters to a single matrix
    tmp = np.concatenate([X, Y, W, H], axis=0)
    tmp = tmp.reshape((1, num_anchors*4, feature_map_size, feature_map_size))
    anchors_tensor = np.repeat(tmp, batch_size, axis=0)
    return anchors_tensor


def get_anchors_centers(original_img_size, feature_map_size):
    '''
    Create the center coordinates x, y of the anchors
    :param original_img_size: height or width of the square-shaped image
    :param feature_map_size: height or width of the square-shaped feature map
    :return: matrices containing center coordinates in the image plane for each position of the
             feature map
    '''
    import numpy as np
    receptive_field_size = int(original_img_size/feature_map_size)
    X = np.zeros((feature_map_size,feature_map_size))
    Y = np.zeros((feature_map_size,feature_map_size))
    for row in range(feature_map_size):
        for col in range(feature_map_size):
            Y[row,col] = col*receptive_field_size + int(receptive_field_size/2)
            X[row,col] = row*receptive_field_size + int(receptive_field_size/2)
    return X, Y


def get_anchors_w_and_h(num_anchors, feature_map_size, anchors_scales, anchors_aspect_ratios):
    '''
    Create height and width of the anchors.
    :param num_anchors: number of anchors per position in the feature map
    :param feature_map_size: size of the square-shaped feature map
    :param anchors_scales: scales of the anchors
    :param anchors_aspect_ratios: aspect ratios of the anchors
    :return: matrices containing width and height of the anchors
    '''
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
