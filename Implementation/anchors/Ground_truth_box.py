class Ground_truth_box:
    '''
    Ground truth boxes are used to evaluate the best fit ground truth box during training, however, the main coordinate
    handling is done by multidimensional matrices (see also Anchor.py for a more detailed description).
    '''

    def __init__(self, x, y, w, h, label):
        '''
        Initializer
        :param x: center x coordinate of the ground truth box
        :param y: center y coordinate of the ground truth box
        :param w: width of the ground truth box
        :param h: height of the ground truth box
        :param label: mnist number class the ground truth box represents
        '''
        # initially set
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        # used by anchor evaluation to assign anchors to ground truth boxes
        self.anchors = []
        self.ious = []

    def add_anchor(self, anchor, iou):
        '''
        Adds an anchor to the ground truth box
        :param anchor: anchor object
        :param iou: IoU between anchor and ground truth box
        :return: nothing, but assigns anchor and IoU to the respective lists
        '''
        self.anchors.append(anchor)
        self.ious.append(iou)
