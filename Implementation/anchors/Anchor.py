class Anchor(object):

    def __init__(self, x, y, w, h, w_idx, h_idx, anchor_idx, lower_threshold, upper_threshold):
        '''
        The anchor objects are used to ease the identification of the optimal ground truth box (e.g. representing an
        mnist image) assigned to the respective anchor and to initially classify the anchor as positive, negative or
        neutral. Though, for the handling of anchor coordinates and the predicted coordinates of the RoIs is mainly
        done by multidimensional matrices / tensors to ease the procedure in tensorflow. Only during training
        the optimal ground truth box is assigned to the anchor for regression.
        :param x: x coordinate of the center of the anchor in the collage, where for even width anchors the 'center
                  pixel' is defined as the pixel at position ((w/2) - 1)
        :param y: y coordinate of the center of the anchor in the collage, where for even width anchors the 'center
                  pixel' is defined as the pixel at position ((h/2) - 1)
        :param w: width of the anchor
        :param h: height of the anchor
        :param w_idx: x position of the anchor in the feature map
        :param h_idx: y position in the feature map
        :param anchor_idx: goes from 0 to (NUM_ANCHORS-1), e.g. from 0 to 8
        :param lower_threshold: if the anchor has no IoU >= lower_threshold its type will be negative
        :param upper_threshold: if the anchor´s  highest IoU with a ground truth box is between the lower and the upper
                                threshold it will be neutral, if it exceeds the higher threshold it will be positive

        '''
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        # The following two variables will be used later to build a ground truth tensor matching the shape of the
        # anchor tensor and prediction tensor
        self.w_idx = w_idx
        self.h_idx = h_idx
        self.anchor_idx = anchor_idx
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        # the following variables serve to determine the optimal ground truth box to assign to the anchor if such a box
        # exists
        self.groundTruthBoxes = []
        self.intersectionsOfUnions = []
        self.assigned_ground_truth_box = None
        self.assigned_iou = 0
        self.type = 'negative'

    def append_ground_truth_box(self, ground_truth_box):
        '''
        Appends ground truth box to anchor only if the IoU is higher than zero.
        :param ground_truth_box: ground truth box object
        :return: nothing, but appends ground truth boxes with IoU > 0 to the anchors respective list and also the IoU
        '''
        iou = self.calculate_iou(ground_truth_box)
        if iou > 0:
            ground_truth_box.add_anchor(self, iou)
            self.intersectionsOfUnions.append(iou)
            self.groundTruthBoxes.append(ground_truth_box)

    def calculate_iou(self, ground_truth_box):
        '''
        Calculates the intersection over union of anchor and ground truth box
        :param ground_truth_box: ground truth box object
        :return: intersection over union
        '''
        import numpy as np
        # calculate most upper-left (x1, y1) and bottom-right (x2, y2) pixel coordinates of
        # anchor and ground truth box
        if self.w % 2 == 0:
            x1_a = self.x - ((self.w / 2) - 1)
            x2_a = self.x + (self.w / 2)
            y1_a = self.y - ((self.h / 2) - 1)
            y2_a = self.y + (self.h / 2)
        else:
            dw = np.floor(self.w/2)
            dh = np.floor(self.h/2)
            x1_a = self.x - dw
            x2_a = self.x + dw
            y1_a = self.y - dh
            y2_a = self.y + dh
        x_t, y_t, w_t, h_t = ground_truth_box.x, ground_truth_box.y, ground_truth_box.w, ground_truth_box.h
        if w_t % 2 == 0:
            x1_t = x_t - (w_t/2 - 1)
            x2_t = x_t + (w_t/2)
            y1_t = y_t - (h_t/2 - 1)
            y2_t = y_t + (h_t/2)
        else:
            dw = np.floor(w_t / 2)
            dh = np.floor(h_t / 2)
            x1_t = x_t - dw
            x2_t = x_t + dw
            y1_t = y_t - dh
            y2_t = y_t + dh
        # only if the anchor and ground truth box actually intersect the IoU is nonzero and can be calculated using
        # the following formula
        if ~(x1_t > x2_a or x1_a > x2_t or y1_t > y2_a or y1_a > y2_t): #
            # calculate coordinates of intersection rectangle
            x1_i = max(x1_a, x1_t)
            y1_i = max(y1_a, y1_t)
            x2_i = min(x2_a, x2_t)
            y2_i = min(y2_a, y2_t)
            # areas of anchor and ground truth box and intersection
            area_a = self.w * self.h
            area_t = w_t * h_t
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            # calculate and return intersection over union
            return intersection / (area_a + area_t - intersection)
        else:
            return 0

    def evaluate_anchor(self):
        if len(self.intersectionsOfUnions) is not 0:
            # only if at least one ground truth box was assigned to the anchor its type can be non-negative
            for b in self.groundTruthBoxes:
                # for every ground truth box assigned to the anchor
                # get the IoU between box and anchor, highest IoU box has with an anchor, highest IoU anchor has with
                # a box and whether the box is the anchor´s best box and the anchor the box´s best anchor
                iuo = self.intersectionsOfUnions[self.groundTruthBoxes.index(b)]
                max_iou_box = max(b.ious)
                max_iou_anchor = max(self.intersectionsOfUnions)
                best_box = (b == self.groundTruthBoxes[
                    self.intersectionsOfUnions.index(max(self.intersectionsOfUnions))])
                best_anchor = (self == b.anchors[b.ious.index(max(b.ious))])
                if max_iou_box < self.upper_threshold and best_anchor:
                    # special case: if a ground truth box exists where no anchor has an IoU >= 0.70 with the box
                    #               and the anchor at hand is from the perspective of the ground truth box the anchor
                    #               with the highest IoU then assign the ground truth box to the anchor at hand
                    self.assigned_ground_truth_box = b
                    self.assigned_iou = iuo
                    self.type = 'positive'
                elif iuo > self.upper_threshold and best_box:
                    # else if the anchor has at least 0.70 with the box and the box is the box the anchors
                    # has the highest IoU with assign the box to the anchor and mark the anchor as positive
                    self.assigned_ground_truth_box = b
                    self.assigned_iou = iuo
                    self.type = 'positive'
                elif self.lower_threshold <= max_iou_anchor < self.upper_threshold and best_box:
                    # if this is the box the anchor has the highest IoU with which is 0.03 <= IoU < 0.70 mark the anchor
                    # as neutral
                    self.type = 'neutral'
                    self.assigned_iou = iuo

                elif max_iou_anchor < self.lower_threshold and best_box:
                    # if the box is the box the anchor has the highest IoU with but it is below 0.70 mark the anchor to
                    # be negative
                    self.type = 'negative'
                    self.assigned_iou = iuo
        else:
            # if no box was assigned to the anchor
            self.type = 'negative'
