'''Data generation: (-> Yan)
- Multi purpose function for putting small mnist images with various scales onto big background frame
- Degree of small mnist images overlap as parameter of script
- Background first in main background color of mnist images
- Different rotation, scaling, ...
- not in tensorflow'''
%matplotlib inline
#from pdb import set_trace as st
import pdb
import matplotlib.pyplot as plt

# TODO: CREATE CORRECT LABELS
# TODO: ADDITIONALLY SAVE ROTATION ANGLE AND SCALE
# TODO: IF DATA IS SAVED SAVE TRAINING SET NAME IN FILENAME
# TODO: REPLACE REDUCE BY CONTINUOUS FUNCTION
# TODO: CALCULATE DATA OVERNIGHT
# TODO: UPLOAD ON GITHUB

def convert_targets_to_bboxes(target):
    pass

def get_mnist():
    '''
    Loads mnist data.
    :return: train images and labels, validation images and labels, test images and labels
    '''
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/")
    train_imgs = mnist.train.images # 10000
    train_labels = mnist.train.labels
    valid_imgs = mnist.validation.images # 55000
    valid_labels = mnist.validation.labels
    test_imgs  = mnist.test.images # 5000
    test_labels = mnist.test.labels
    return train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels

train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels = get_mnist()

def create_collages(num_collages=15, collage_size=128, min_num_imgs=1, max_num_imgs=10, replacement=True,
                    allow_overhang=True, background='black',
                    num_down_scales=2, num_up_scales=1,
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

    # get rotated and scaled mnist images
    mnist_transformed = []
    for dataset in [train_imgs, valid_imgs, test_imgs]:
        mnist_transformed.append(
            augment_mnist(dataset, counterclock_angle=counterclock_angle, clockwise_angle=clockwise_angle,
                          rotation_steps=rotation_steps, num_down_scales=num_down_scales, num_up_scales=num_up_scales))

    # train labels shape: (55000,)
    mnist_labels = []
    for dataset in [train_labels, valid_labels, test_labels]:
        mnist_labels.append(np.repeat(dataset,rotation_steps*(num_up_scales+num_down_scales+1)))

    # Create collages of mnist images and corresponding labels
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
            drawn_targets = mnist_labels[ds_indx][rand_indxs]


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
                    #print('x1:',x1,'x2:',x2,'y1:',y1,'y2:',y2,'\n','i1:',i1,'i2:',i2,'j1:',j1,'j2:',j2, 'img[1].shape:', img[1].shape,'xindx:',xindx,'yindx',yindx, 'tmp_i1:',tmp_i1,'tmp_i2:',tmp_i2,'tmp_j1:',tmp_j1,'tmp_j2', tmp_j2)
                    collage_frame[x1:x2,y1:y2] = img[1][i1:i2,j1:j2]


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
                    collage_frame[xindx:xindx+s,yindx:yindx+s] = img[1]
                    # prepare label including number, position x, y and width and height
                    x = int(xindx+(s/2))
                    y = int(yindx+(s/2))
                    h, w = s, s

                # for each added mnist image add list of mnist number, position and height and width to the list
                # for the targets of the respective collage
                targets[ds_indx][c].append([drawn_targets[img[0]], x, y, h, w])

            # same dataset but finished collage here...
            plt.imshow(collage_frame)
            plt.show()
            collages[ds_indx].append(collage_frame)
            print(targets[ds_indx][c])


    return collages

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

def augment_mnist(imgs, counterclock_angle=30, clockwise_angle=30, rotation_steps=7, num_down_scales=2, num_up_scales=2):
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
    # check whether this set of augmented images was already created:
    import os
    import numpy as np
    file_name = '' + str(counterclock_angle) + '_' + str(clockwise_angle) + '_' + str(rotation_steps) + '_' + \
                str(num_down_scales) + '_' + str(num_up_scales) + '.pkl'
    existence = os.path.isfile(file_name)
    # if the data set does not already exist create it
    if existence == False:
        augmented_imgs = []
        for i1 in range(imgs.shape[0]):
            seq = rotation_sequence(imgs[i1], counterclock_angle, clockwise_angle, rotation_steps)
            for i2 in seq:
                tmp = scale_pyramid(i2, num_down_scales, num_up_scales)
                for i3 in tmp:
                    augmented_imgs.append(i3)
            print(i1,'/',imgs.shape[0])
        # save dataset to disk
        import pickle
        with open(file_name, 'wb') as f:
            pickle.dump(augmented_imgs, f)
    else:
        # load already existing data set
        import pickle
        with open(file_name, 'rb') as file:
            augmented_imgs = pickle.load(file)
    return np.array(augmented_imgs)

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
    angles = list(np.linspace(-counterclock_angle,clockwise_angle,steps)[::-1])
    for a in angles:
        if a < 0:
            seq.append(img)
        seq.append(rotate(img, a))
    return seq

def scale_pyramid(img, num_down_scales=2, num_up_scales=2):
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
    from skimage.transform import pyramid_reduce as reduce
    from skimage.transform import pyramid_expand as expand
    if img.shape != (28,28):
        img = img.reshape((28,28))
    pyramid = []
    for i in range(num_up_scales,0,-1):
        tmp = expand(img, 2**i)
        pyramid.append(tmp)
    pyramid.append(img)
    for i in range(1,num_down_scales+1,1):
        tmp = reduce(img, 2**i)
        pyramid.append(tmp)
    return pyramid

# For testing:
#import matplotlib.pyplot as plt
#create_collages()
train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels = get_mnist()
print(len(test_imgs))
augment_mnist(test_imgs)
#import pickle
#with open('30_30_7_2_2.pkl', 'rb') as file:
#    res = pickle.load(file)
#for i in res:
#    plt.imshow(i)
#    plt.show()