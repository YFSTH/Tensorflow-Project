import pdb

class Anchor(object):

    def __init__(self, x, y, w, h, w_idx, h_idx, anchor_idx, lower_threshold, upper_threshold):
        '''

        :param x:
        :param y:
        :param w:
        :param h:
        :param w_idx: x position in the feature map
        :param h_idx: y position in the feature map
        :param anchor_index: goes from 0 to (NUM_ANCHORS-1), e.g. from 0 to 8
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

        self.groundTruthBoxes = []
        self.intersectionsOfUnions = []
        self.assigned_ground_truth_box = None
        self.assigned_iou = 0
        self.type = 'negative'

    def append_ground_truth_box(self, ground_truth_box):
        iou = self.calculate_iou(ground_truth_box)
        if iou > 0:
            ground_truth_box.add_anchor(self, iou)
            self.intersectionsOfUnions.append(iou)
            self.groundTruthBoxes.append(ground_truth_box)

    def calculate_iou(self, ground_truth_box):
        '''
        Calculates the intersection over union of anchor and ground truth box
        :param ground_truth_box: tuple with (class, x, y, w, h, angle, scale)
        :return: intersection over union
        '''
        import numpy as np

        # calculate most upper-left (x1,y1) and bottom-right (x2,y2) pixel coordinates of
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
            x1_a = x_t - dw
            x2_a = x_t + dw
            y1_a = y_t - dh
            y2_a = y_t + dh


        if ~(x1_t > x2_a or x1_a > x2_t or y1_t > y2_a or y1_a > y2_t): # if anchor and ground truth box intersect
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

            upper_threshold = 0.70


            for b in self.groundTruthBoxes:

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
                    # if the anchor has an IoU > 0.70 with the ground truth box and if this box is from the anchors
                    # point of view the box with the highest IoU then assign box to this anchor
                    self.assigned_ground_truth_box = b
                    self.assigned_iou = iuo
                    self.type = 'positive'

                elif self.lower_threshold <= max_iou_anchor < self.upper_threshold and best_box:
                    self.type = 'neutral'
                    self.assigned_iou = iuo

                elif max_iou_anchor < self.lower_threshold and best_box:
                    self.type = 'negative'
                    self.assigned_iou = iuo

        else:
            self.type = 'negative'


