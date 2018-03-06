def get_mnist():
    """
    Loads mnist data.
    :return: train images and labels, validation images and labels, test images and labels
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/")
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    valid_imgs = mnist.validation.images
    valid_labels = mnist.validation.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels
    return train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels


def create_collages(num_collages=5, collage_size=128, min_num_imgs=1, max_num_imgs=3, replacement=True,
                    allow_overhang=True, background='black', min_scaling=0.5, max_scaling=1.5, scaling_steps=2,
                    counterclock_angle=0, clockwise_angle=0, rotation_steps=2):
    '''
    Create collage images contain a certain number of mnist images
    :param num_collages: number of collages that shall be created
    :param collage_size: size of the square shaped collage (size = width = height)
    :param min_num_imgs: minimum number of mnist images to place on each collage
    :param max_num_imgs: maxium number of mnist images to place one ach collage
    :param replacement: whether a single mnist image of a specific scale and rotation angle can be drawn again
    :param allow_overhang: whether parts of the mnist image are allowed to be outside of the collage frame
    :param background: background of the collage frame (black, white or gaussian noise)
    :param min_scaling: minimum scaling of the single mnist images
    :param max_scaling: maximum scaling of the single mnist images
    :param scaling_steps: steps from min. to max. scaling of mnist images
    :param counterclock_angle: max. counterclockwise rotation of the mnist images
    :param clockwise_angle: max. clockwise rotation of the mnist images
    :param rotation_steps: steps from max. counerclockwise to clockwise rotation angle
    :return: nothing, but save the collages to the disk
    '''
    import numpy as np
    import os
    # Check whether collages already exist if that is the case do nothing
    collages_filename = str(num_collages)+'_'+str(collage_size)+'_'+str(min_num_imgs)+'_'+str(max_num_imgs)+'_'+str(replacement)+'_'+str(allow_overhang)+'_'+str(background)+'_'+str(min_scaling)+'_'+str(max_scaling)+'_'+str(scaling_steps)+'_'+str(counterclock_angle)+'_'+str(clockwise_angle)+'_'+str(rotation_steps)
    targets_filename  = str(num_collages)+'_'+str(collage_size)+'_'+str(min_num_imgs)+'_'+str(max_num_imgs)+'_'+str(replacement)+'_'+str(allow_overhang)+'_'+str(background)+'_'+str(min_scaling)+'_'+str(max_scaling)+'_'+str(scaling_steps)+'_'+str(counterclock_angle)+'_'+str(clockwise_angle)+'_'+str(rotation_steps)
    files = os.listdir('./data_generation')
    existence = False
    for f in files:
        if collages_filename in f or targets_filename in f:
            existence = True
    if existence == False:
        # if the collages do not yet exist
        # get raw mnist images and labels
        train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels = get_mnist()
        # draw a subset of the total number of mnist images
        rnd_indxs = np.random.choice(np.arange(0,len(train_imgs)), 15000, replace=False)
        train_imgs_selection = train_imgs[rnd_indxs]
        train_labels_selection = train_labels[rnd_indxs]
        # get rotated and scaled mnist images
        mnist_transformed, mnist_labels, angles, scales = [], [], [], []
        for dataset in [(valid_imgs,'validation', valid_labels), (train_imgs_selection,'training', train_labels_selection),(test_imgs,'test', test_labels)]:
            # augment the mnist images by applying different rotations and scales
            transformed, labels, angles_, scales_ = augment_mnist(dataset[0], dataset[1], dataset[2], counterclock_angle=counterclock_angle, clockwise_angle=clockwise_angle,
                                                          rotation_steps=rotation_steps, min_scaling=min_scaling, max_scaling=max_scaling, scaling_steps=scaling_steps)
            mnist_transformed.append(transformed)
            angles.append(angles_)
            scales.append(scales_)
            mnist_labels.append(labels)
        # for storage create a collage and a target superlist containing three sublists for the training, validation,
        # and test dataset respectively
        collages = [[], [], []]
        targets =  [[], [], []]
        for dataset in enumerate(mnist_transformed):
            # for each dataset (training, validation or testing)
            ds_indx = dataset[0]
            for c in range(num_collages):
                # for every collage that shall be created
                # create a collage background frame
                collage_frame = create_frame(collage_size, background)
                # randomly choose index of mnist images for collage
                num_mnist_imgs = np.random.randint(min_num_imgs, max_num_imgs+1)
                rand_indxs = list(np.random.choice(np.arange(0, len(dataset[1])), num_mnist_imgs, replace=replacement))
                # get the image, angle, scale and label belonging to the randomly drawn index
                drawn_imgs = dataset[1][rand_indxs]
                drawn_angles = angles[dataset[0]][rand_indxs]
                drawn_scales = scales[dataset[0]][rand_indxs]
                drawn_targets = mnist_labels[dataset[0]][rand_indxs]
                # add label list for every collage that is created
                targets[ds_indx].append([])
                # randomly place them in frame according to specifications
                for img in enumerate(drawn_imgs):
                    # index if full image can be added to collage
                    i1, i2, j1, j2 = 0, img[1].shape[0], 0, img[1].shape[1]
                    if allow_overhang:
                        # get mnist image shape, square shape is assumed
                        n, m = img[1].shape
                        # half width/height of image (images are square)
                        size = int(np.floor(img[1].shape[0] / 2))
                        # randomly draw indices from the whole collage frame
                        xindx = np.random.randint(0, collage_size)
                        yindx = np.random.randint(0, collage_size)
                        if n % 2 == 0:
                            # if the image e.g. width is even
                            # get top-left i1 and bottom-right i2 x coordinate of mnist image on frame
                            tmp_i1 = xindx - (size - 1)
                            tmp_i2 = (collage_frame.shape[0] - 1) - (xindx + size)
                            # get top-left j1 and bottom-right j2 y coordinate
                            tmp_j1 = yindx - (size - 1)
                            tmp_j2 = (collage_frame.shape[1] - 1) - (yindx + size)
                        else:
                            # if the image e.g. width is odd
                            tmp_i1 = xindx - size
                            tmp_i2 = (collage_frame.shape[0] - 1) - (xindx + size)
                            tmp_j1 = yindx - size
                            tmp_j2 = (collage_frame.shape[1] - 1) - (yindx + size)
                        # shrink ranges of indices of mnist image (so only the part of the mnist image is used, which does
                        # not overhang)
                        if tmp_i1 < 0:
                            # if there is overhang in the x-direction
                            # shrinked top-left x-coordinate i1 of mnist image
                            i1 = i1 + abs(tmp_i1)
                        if tmp_i2 < 0:
                            # shrinked bottom-right x-coordinate i2 of mnist image
                            i2 = (img[1].shape[0]) - abs(tmp_i2)
                        if tmp_j1 < 0:
                            # shrinked top-left y coordinate j1 of the mnist image
                            j1 = j1 + abs(tmp_j1)
                        if tmp_j2 < 0:
                            # shrinked bottom-right y coordninate j2 of the mnist image
                            j2 = (img[1].shape[1]) - abs(tmp_j2)
                        # get to-be-used subarray coordinates of collage frame
                        if n % 2 == 0:
                            x1 = xindx - (size - 1) + abs(tmp_i1) * (tmp_i1<0)
                            x2 = xindx + size + 1 - abs(tmp_i2) * (tmp_i2<0)
                            y1 = yindx - (size - 1) + abs(tmp_j1) * (tmp_j1 < 0)
                            y2 = yindx + size + 1 - abs(tmp_j2) * (tmp_j2 < 0)
                        else:
                            x1 = xindx - (size) + abs(tmp_i1) * (tmp_i1 < 0)
                            x2 = xindx + size + 1 - abs(tmp_i2) * (tmp_i2 < 0)
                            y1 = yindx - (size) + abs(tmp_j1) * (tmp_j1 < 0)
                            y2 = yindx + size + 1 - abs(tmp_j2) * (tmp_j2 < 0)
                        # draw mnist image onto collage frame
                        collage_frame[x1:x2, y1:y2] += img[1][i1:i2, j1:j2]
                        # prepare label including position x, y and width and height
                        x = xindx
                        y = yindx
                        h = i2 - i1 + 1
                        w = j2 - j1 + 1
                    else:
                        # if overhang is not allowed  determine the allowed placement positions for the first and
                        # second dimension (in contrast to the above case does (xindx,yindx) represent the upper-left
                        # pixel where the mnist image will be drawn onto the collage
                        s = img[1].shape[0]
                        xmax = collage_size - s
                        ymax = collage_size - s
                        # randomly draw indices from the whole collage frame
                        xindx = np.random.randint(0, xmax + 1)
                        yindx = np.random.randint(0, ymax + 1)
                        # draw mnist image onto collage frame
                        collage_frame[xindx:xindx+s,yindx:yindx+s] += img[1]
                        # prepare label including number, position x, y and width and height
                        x = int(xindx+(s/2)-1)
                        y = int(yindx+(s/2)-1)
                        h, w = s, s
                    # for each added mnist image add list of mnist number, position and height and width to the list
                    # for the targets of the respective collage
                    targets[ds_indx][c].append([drawn_targets[img[0]], x, y, h, w, drawn_angles[img[0]], drawn_scales[img[0]]])
                # append created collage to list of collages created for the dataset
                collages[ds_indx].append(collage_frame)
                # if target list is to short fill it up with placeholder tokens (necessary for use in tensorflow)
                while len(targets[ds_indx][c]) < max_num_imgs:
                    targets[ds_indx][c].append([-9, -9, -9, -9, -9, -9, -9])
            # identify dataset
            if dataset[0] == 0:
                    name = 'train'
            if dataset[0] == 1:
                name = 'test'
            if dataset[0] == 2:
                name = 'valid'
            # sace collages and collage targets
            collages_filename = str(num_collages) + '_' + str(collage_size) + '_' + str(min_num_imgs) + '_' + \
                                str(max_num_imgs) + '_' + str(replacement) + '_' + str(allow_overhang) + '_' + \
                                str(background) + '_' + str(min_scaling) + '_' + str(max_scaling) + '_' + \
                                str(scaling_steps) + '_' + str(counterclock_angle) + '_' + str(clockwise_angle) + \
                                '_' + str(rotation_steps) + '_' + name + '_' + 'collages.pkl'
            targets_filename = str(num_collages) + '_' + str(collage_size) + '_' + str(min_num_imgs) + \
                               '_' + str(max_num_imgs) + '_' + str(replacement) + '_' + str(allow_overhang) + \
                               '_' + str(background) + '_' + str(min_scaling) + '_' + str(max_scaling) + '_' + \
                               str(scaling_steps) + '_' + str(counterclock_angle) + '_' + str(clockwise_angle) + '_' + \
                               str(rotation_steps) + '_' + name + '_' + 'targets.pkl'
            import pickle
            with open('data_generation/'+collages_filename, 'wb') as f:
                    pickle.dump(collages[dataset[0]], f)
            with open('data_generation/'+targets_filename, 'wb') as f:
                    pickle.dump(targets[dataset[0]], f)
            print(str(name)+' augmented mnist collages were created')


def create_frame(frame_size, background):
    """
    Creates a background frame for the mnist collages
    :param frame_size: size of the square shaped mnist collage
    :param background: 'black', 'white' or 'gaussian_noise'
    :return: frame of specified size and background
    """
    import numpy as np
    if background == 'black':
        return np.zeros((frame_size, frame_size))
    elif background == 'white':
        return np.ones((frame_size, frame_size))
    else:
        # gaussian noise
        tmp = abs(np.random.normal(0,1,(frame_size, frame_size)))
        return tmp / np.max(tmp)


def augment_mnist(imgs, name, labels, counterclock_angle=30, clockwise_angle=30, rotation_steps=7, min_scaling=0.25,
                  max_scaling=2.0, scaling_steps=5):
    '''
    Create set of augmented mnist images with the specified rotation angles and scales.
    :param imgs: image set
    :param name: file name postfix, e.g. train, validation, test
    :param labels: label set belonging to the image
    :param counterclock_angle: maximum counterclockwise rotation
    :param clockwise_angle: maximum clockwise rotation
    :param rotation_steps: number of steps from max. counterclockwise to max clockwise rotation
    :param min_scaling: min. scaling factor of image
    :param max_scaling: max. scaling factor of image
    :param scaling_steps: number of steps from min. to max. scaling
    :return: sequence of augmented mnist images containing each combination of rotation and scaling variant,
             the label for each of the mnist images (because the output number of mnist images is higher than the input
             number this is required), sequence of rotation angles and sequence of scales corresponding to the sequence
             of the augmented mnist images
    '''
    import numpy as np
    # repeat label of a single image to match the number of replications of this mnist image
    labels = np.repeat(labels, rotation_steps*scaling_steps)
    # check whether this set of augmented images was already created
    import os
    import numpy as np
    base_name = '' + str(counterclock_angle) + '_' + str(clockwise_angle) + '_' + str(rotation_steps) + '_' + \
                str(min_scaling) + '_' + str(max_scaling) + '_' + str(scaling_steps) + '_' + name
    file1_name = base_name + '.pkl'
    file2_name = base_name + '_angles.pkl'
    file3_name = base_name + '_scales.pkl'
    file4_name = base_name + '_labels.pkl'
    existence = os.path.isfile(file1_name)
    import pickle
    if existence == False:
        # if the data set does not already exist create it
        augmented_imgs = []
        scales = []
        angles = []
        for i1 in range(imgs.shape[0]):
            # for every mnist image apply the rotation sequence
            seq, angls = rotation_sequence(imgs[i1], counterclock_angle, clockwise_angle, rotation_steps)
            for i2 in enumerate(seq):
                # for every created image of the rotation sequence apply the scaling sequence
                tmp, scals = scale_pyramid(i2[1], min_scaling=min_scaling, max_scaling=max_scaling,
                                           scaling_steps=scaling_steps)
                # generate angle labels
                angls_ = np.repeat(angls[i2[0]], len(tmp))
                for i3, scale, angle in zip(tmp, scals, angls_):
                    # save the generated mnist image
                    augmented_imgs.append(i3)
                    # ... and the scale and angle
                    scales.append(scale)
                    angles.append(angle)

        # save dataset to disk
        with open('data_generation/'+file1_name, 'wb') as f:
            pickle.dump(augmented_imgs, f)
        with open('data_generation/'+file2_name, 'wb') as f:
            pickle.dump(angles, f)
        with open('data_generation/'+file3_name, 'wb') as f:
            pickle.dump(scales, f)
        with open('data_generation/'+file4_name, 'wb') as f:
            pickle.dump(labels, f)
        print('augmented ' + str(name) + ' mnist images were created')
    else:
        # load already existing data set
        import pickle
        with open('data_generation/'+file1_name, 'rb') as file:
            augmented_imgs = pickle.load(file)
        with open('data_generation/'+file2_name, 'rb') as file:
            angles = pickle.load(file)
        with open('data_generation/'+file3_name, 'rb') as file:
            scales = pickle.load(file)
        with open('data_generation/'+file4_name, 'rb') as file:
            labels = pickle.load(file)
        print(str(name) + ' augmented mnist images were loaded')
    return np.array(augmented_imgs), np.array(labels), np.array(angles), np.array(scales)


def rotation_sequence(img, counterclock_angle=30, clockwise_angle=30, steps=7):
    """
    Rotate the image starting with a counter-clockwise rotation and proceed with rotations going more and more into
    the clockwise rotation direction in the defined number of steps.
    :param img: input gray value image
    :param counterclock_angle: initial counter-clockwise rotation angle, must be positive number
    :param clockwise_angle: last clockwise rotation angle, must be positive number
    :param steps: number of computed angles between counterclock-wise and clockwise rotation angle
    :return: list of rated images, starting with the counter-clockwise rotated
    """
    from skimage.transform import rotate
    import numpy as np
    if img.shape != (28, 28):
        img = img.reshape((28,28))
    seq = []
    angles = list(np.linspace(counterclock_angle,-clockwise_angle,steps))
    for a in angles:
        # apply skimage rotation operation
        seq.append(rotate(img, a))
    return seq, angles


def scale_pyramid(img, min_scaling=0.25, max_scaling=2.0, scaling_steps=5):
    """
    Create a scale pyramid of image. Thus create upscaled and downscaled ones.
    :param img: input gray value image
    :param min_scaling: minimum scaling factor
    :param max_scaling: maximum scaling factor
    :return: list with image in different scales (e.g. 2x, 1x, 0.5x)
    """
    from skimage.transform import rescale
    import numpy as np
    if img.shape != (28,28):
        # if the mnist images were saved in 1D shape instaed of 2D
        img = img.reshape((28,28))
    pyramid = []
    scales = np.linspace(min_scaling, max_scaling, scaling_steps)
    for scale in scales:
        # apply skimage scaling
        pyramid.append(rescale(img, scale))
    return pyramid, scales

