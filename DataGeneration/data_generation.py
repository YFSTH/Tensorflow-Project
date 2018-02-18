%matplotlib inline
import matplotlib.pyplot as plt
import pdb

# TODO: UPLOAD ON GITHUB

def get_mnist():
    '''
    Loads mnist data.
    :return: train images and labels, validation images and labels, test images and labels
    '''
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/")
    train_imgs = mnist.train.images # 55000
    train_labels = mnist.train.labels
    valid_imgs = mnist.validation.images # 5000
    valid_labels = mnist.validation.labels
    test_imgs  = mnist.test.images # 10000
    test_labels = mnist.test.labels
    return train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels

train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels = get_mnist()

def create_collages(num_collages=15, collage_size=128, min_num_imgs=2, max_num_imgs=8, replacement=True,
                    allow_overhang=True, background='black',
                    min_scaling=0.25, max_scaling=2.0, scaling_steps=5,
                    counterclock_angle=30, clockwise_angle=30, rotation_steps=7):
    '''
    Creates images (collages) containing a certain number of mnist images which exhibit a specified overlap.
    :param imgs: mnist train, validation xor test set
    :param labels: mnist train, validation xor test labels
    :param num_collages: number of to-be-created collages
    :param collage_size: frame size of the collage image
    :param min_num_imgs: minimum number of mnist images in each collage
    :param max_num_imgs: maximum number of mnist images in each collage
    :param allow_overhang: whether the small mnist images in the collage can partly be outside the collage frame
    :background: gray value of the background or whether gaussian noise shall be applied
    :param replacement: whether it is allowed to put a single (rotated and scaled) mnist image on multiple collages
                        (=drawing with replacement from the pool of rotated and scaled mnist images)
    :return: the set of created collages containing multiple mnist images on a bigger frame and a label dictionary
             characterized by: {'target_numbers': [t1,...], 'target_positions': [(x1,y1),...]}
    '''
    import numpy as np

    # get raw mnist images and labels
    train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels = get_mnist()
    
    rnd_indxs = np.random.choice(np.arange(0,len(train_imgs)), 15000, replace=False)
    train_imgs_selection = train_imgs[rnd_indxs]
    train_labels_selection = train_labels[rnd_indxs]
    
    # get rotated and scaled mnist images
    mnist_transformed, mnist_labels, angles, scales = [], [], [], []
    
    for dataset in [(train_imgs_selection,'training', train_labels_selection),(test_imgs,'test', test_labels), (valid_imgs,'validation', valid_labels)]:
        transformed, labels, angles_, scales_ = augment_mnist(dataset[0], dataset[1], dataset[2], counterclock_angle=counterclock_angle, clockwise_angle=clockwise_angle,
                                                      rotation_steps=rotation_steps, min_scaling=min_scaling, max_scaling=max_scaling, scaling_steps=scaling_steps)
        mnist_transformed.append(transformed)
        angles.append(angles_)
        scales.append(scales_)     
        mnist_labels.append(labels)
    
    collages = [[], [], []]
    targets =  [[], [], []]

    for dataset in enumerate(mnist_transformed):
        ds_indx = dataset[0]

        for c in range(num_collages):
            collage_frame = create_frame(collage_size, background)
            
                
            # randomly choose random number of mnist images for collage
            num_mnist_imgs = np.random.randint(min_num_imgs, max_num_imgs+1)
            rand_indxs = list(np.random.choice(np.arange(0, len(dataset[1])), num_mnist_imgs, replace=replacement))
            drawn_imgs = dataset[1][rand_indxs]
            drawn_angles = angles[dataset[0]][rand_indxs]
            drawn_scales = scales[dataset[0]][rand_indxs]
            drawn_targets = mnist_labels[dataset[0]][rand_indxs]
            
            # add label list for every collage that is created
            targets[ds_indx].append([])

            # randomly place them in frame according to specifications
            for img in enumerate(drawn_imgs):

                # only implemented for square shaped images
                assert(img[1].shape[0] == img[1].shape[1])

                # index if full image can be added to collage
                i1, i2, j1, j2 = 0, img[1].shape[0], 0, img[1].shape[1]

                if allow_overhang:
                    # get mnist image shape
                    n, m = img[1].shape

                    # half width/height of image (images are square)
                    size = int(np.floor(img[1].shape[0] / 2))

                    # randomly draw indices from the whole collage frame
                    xindx = np.random.randint(0, collage_size)
                    yindx = np.random.randint(0, collage_size)
                    
                    if n % 2 == 0:
                        tmp_i1 = xindx - (size - 1)
                        tmp_i2 = (collage_frame.shape[0] - 1) - (xindx + size)
                        tmp_j1 = yindx - (size - 1)
                        tmp_j2 = (collage_frame.shape[1] - 1) - (yindx + size)
                    else:
                        tmp_i1 = xindx - size
                        tmp_i2 = (collage_frame.shape[0] - 1) - (xindx + size)
                        tmp_j1 = yindx - size
                        tmp_j2 = (collage_frame.shape[1] - 1) - (yindx + size)

                    # shrink ranges of indices of mnist image (so only the part of the mnist image is used, which does
                    # not overhang)
                    if tmp_i1 < 0:
                        i1 = i1 + abs(tmp_i1)
                    if tmp_i2 < 0:
                        i2 = (img[1].shape[0]) - abs(tmp_i2)
                    if tmp_j1 < 0:
                        j1 = j1 + abs(tmp_j1)
                    if tmp_j2 < 0:
                        j2 = (img[1].shape[1]) - abs(tmp_j2)

                    # get to-be-used subarray of collage frame
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
                    collage_frame[x1:x2,y1:y2] += img[1][i1:i2,j1:j2]


                    # prepare label including number, position x, y and width and height
                    #number = labels[]
                    x = xindx
                    y = yindx
                    h = i2 - i1 + 1
                    w = j2 - j1 + 1

                else:
                    # determine the allowed placement positions for the first and second dimension (in contrast to the above
                    # case does (xindx,yindx) represent the upper-left pixel where the mnist image will be drawn onto
                    # the collage
                    s = img[1].shape[0]
                    xmax = collage_size - s
                    ymax = collage_size - s
                    
                    # randomly draw indices from the whole collage frame
                    xindx = np.random.randint(0, xmax + 1)
                    yindx = np.random.randint(0, ymax + 1)
                    
                    # draw mnist image onto collage frame
                    collage_frame[xindx:xindx+s,yindx:yindx+s] += img[1]
                    
                    # prepare label including number, position x, y and width and height
                    x = int(xindx+(s/2))
                    y = int(yindx+(s/2))
                    h, w = s, s

                # for each added mnist image add list of mnist number, position and height and width to the list
                # for the targets of the respective collage
                targets[ds_indx][c].append([drawn_targets[img[0]], x, y, h, w, drawn_angles[img[0]], drawn_scales[img[0]]])

            collages[ds_indx].append(collage_frame)
            
            plt.imshow(collage_frame)
            plt.show()
            print(targets[ds_indx][c])        
            
    import pickle
    collages_filename = str(num_collages)+'_'+str(collage_size)+'_'+str(min_num_imgs)+'_'+str(max_num_imgs)+'_'+str(replacement)+'_'+str(allow_overhang)+'_'+str(background)+'_'+str(min_scaling)+'_'+str(max_scaling)+'_'+str(scaling_steps)+'_'+str(counterclock_angle)+'_'+str(clockwise_angle)+'_'+str(rotation_steps)+'collages.pkl'
    targets_filename  = str(num_collages)+'_'+str(collage_size)+'_'+str(min_num_imgs)+'_'+str(max_num_imgs)+'_'+str(replacement)+'_'+str(allow_overhang)+'_'+str(background)+'_'+str(min_scaling)+'_'+str(max_scaling)+'_'+str(scaling_steps)+'_'+str(counterclock_angle)+'_'+str(clockwise_angle)+'_'+str(rotation_steps)+'targets.pkl'    
    with open(collages_filename, 'wb') as f:
            pickle.dump(collages, f)   
    with open(targets_filename, 'wb') as f:
            pickle.dump(targets, f)   

    return collages, targets

def create_frame(frame_size, background):
    '''
    Creates a frame for the creation of mnist collages
    :param frame_size: size of the square shaped mnist collage
    :param background: 'black', 'white' or 'gaussian_noise'
    :return: frame of specified size and background
    '''
    import numpy as np
    if background == 'black':
        return np.zeros((frame_size, frame_size))
    elif background == 'white':
        return np.ones((frame_size, frame_size))
    else:
        tmp = abs(np.random.normal(0,1,(frame_size, frame_size)))
        return (tmp / np.max(tmp))

def augment_mnist(imgs, name, labels, counterclock_angle=30, clockwise_angle=30, rotation_steps=7, min_scaling=0.25, max_scaling=2.0, scaling_steps=5):
    '''
    Create set of mnist images with the specified rotation angles and scales.
    :param imgs: the train, validation xor test mnist image set
    :param counterclock_angle: initial counter-clockwise rotation angle, must be positive number
    :param clockwise_angle: last clockwise rotation angle, must be positive number
    :param rotation_steps: number of computed angles between counterclock-wise and clockwise rotation angle
    :param num_down_scale: number of reduce operations
    :param num_up_scales: number of expand operations
    :return: mnist images with different rotations and scales
    '''
    import numpy as np
    labels = np.repeat(labels, rotation_steps*scaling_steps)
    
    # check whether this set of augmented images was already created:
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
    # if the data set does not already exist create it
    if existence == False:
        augmented_imgs = []
        scales = []
        angles = []
        for i1 in range(imgs.shape[0]):
            seq, angls = rotation_sequence(imgs[i1], counterclock_angle, clockwise_angle, rotation_steps)
            # 7 angles of same img
            
            for i2 in enumerate(seq):
                tmp, scals = scale_pyramid(i2[1], min_scaling=min_scaling, max_scaling=max_scaling, scaling_steps=scaling_steps)
                # gives 5 scales (of each of the seven rotated images)
                #angls_ = np.repeat(angls[i2[0]], len(tmp)) # one rotation angle repeated as often as there are diff scales
                angls_ = np.repeat(angls[i2[0]], len(tmp))
                

                
                for i3, scale, angle in zip(tmp, scals, angls_): # 
                    augmented_imgs.append(i3)
                    scales.append(scale)
                    angles.append(angle)
        
            print(i1,'/',imgs.shape[0])
            
        # save dataset to disk
        with open(file1_name, 'wb') as f:
            pickle.dump(augmented_imgs, f)
        with open(file2_name, 'wb') as f:
            pickle.dump(angles, f)
        with open(file3_name, 'wb') as f:
            pickle.dump(scales, f)
        with open(file4_name, 'wb') as f:
            pickle.dump(labels, f)
    else:
        # load already existing data set
        import pickle
        with open(file1_name, 'rb') as file:
            augmented_imgs = pickle.load(file)
        with open(file2_name, 'rb') as file:
            angles = pickle.load(file)
        with open(file3_name, 'rb') as file:
            scales = pickle.load(file)
        with open(file4_name, 'rb') as file:
            labels = pickle.load(file)
    #for i in range(len(augmented_imgs)): plt.imshow(augmented_imgs[i]);plt.show();print(labels[i])
    return np.array(augmented_imgs), np.array(labels), np.array(angles), np.array(scales)

def rotation_sequence(img, counterclock_angle=30, clockwise_angle=30, steps=7):
    '''
    Rotate the image starting with a counter-clockwise rotation and proceed with rotations going more and more into
    the clockwise rotation direction in the defined number of steps.
    :param img: input gray value image
    :param counterclock_angle: initial counter-clockwise rotation angle, must be positive number
    :param clockwise_angle: last clockwise rotation angle, must be positive number
    :param steps: number of computed angles between counterclock-wise and clockwise rotation angle
    :return: list of rated images, starting with the counter-clockwise rotated
    '''
    from skimage.transform import rotate
    import numpy as np
    if img.shape != (28, 28):
        img = img.reshape((28,28))
    seq = []
    angles = list(np.linspace(counterclock_angle,-clockwise_angle,steps))
    for a in angles:
        seq.append(rotate(img, a))
    
    return seq, angles

def scale_pyramid(img, min_scaling=0.25, max_scaling=2.0, scaling_steps=5):
    '''
    Create a scale pyramid of image. Thus create upscaled representations via expand operation and downscaled ones via
    the reduce operation. Before the up-/down-scaling these operations apply a low-pass filter in the form of a gaussian
    kernel to avoid aliasing artifacts due to the sampling frequency being lower than fourier space frequencies of the
    image.
    :param img: input gray value image
    :param num_down_scales: number of reduce operations
    :param num_up_scales: number of expand operations
    :return: list with image in different scales (e.g. 2x, 1x, 0.5x)
    '''
    from skimage.transform import rescale
    import numpy as np
    if img.shape != (28,28):
        img = img.reshape((28,28))
    pyramid = []
    scales = np.linspace(min_scaling, max_scaling, scaling_steps)
    for scale in scales:
        pyramid.append(rescale(img, scale))
    return pyramid, scales

create_collages()