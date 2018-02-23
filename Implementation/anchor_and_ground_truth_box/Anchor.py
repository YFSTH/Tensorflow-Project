import pdb

class Anchor(object):

    def __init__(self, x, y, w, h, w_idx, h_idx, anchor_idx):
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

        self.groundTruthBoxes = []
        self.intersectionsOfUnions = []
        self.assigned_ground_truth_box = None
        self.assigned_iou = -1
        self.type = 'positive'

    def append_ground_truth_box(self, ground_truth_box):
        iou = self.calculate_iou(ground_truth_box)
        ground_truth_box.add_anchor(self, iou)
        if iou > 0:
            self.intersectionsOfUnions.append(iou)
            self.groundTruthBoxes.append(ground_truth_box)
            self.evaluate_anchor()

    def calculate_iou(self, ground_truth_box):
        '''
        Calculates the intersection over union of anchor and ground truth box
        :param ground_truth_box: tuple with (class, x, y, w, h, angle, scale)
        :return: intersection over union
        '''
        # calculate most upper-left (x1,y1) and bottom-right (x2,y2) pixel coordinates of
        # anchor and ground truth box
        x1_a = self.x - (self.w/2)
        x2_a = self.x + (self.w/2)
        y1_a = self.y - (self.h/2)
        y2_a = self.y + (self.h/2)
        x_t, y_t, w_t, h_t = ground_truth_box.x, ground_truth_box.y, ground_truth_box.w, ground_truth_box.h
        x1_t = x_t - (w_t/2)
        x2_t = x_t + (w_t/2)
        y1_t = y_t - (h_t/2)
        y2_t = y_t + (h_t/2)

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
        for b in self.groundTruthBoxes:
            if max(b.ious) < 0.70 and self == b.anchors[b.ious.index(max(b.ious))]:
                # special case: if a ground truth box exists where no anchor has an IoU >= 0.70 with the box
                #               and the anchor at hand is from the perspective of the ground truth box the anchor
                #               with the highest IoU then assign the ground truth box to the anchor at hand
                self.assigned_ground_truth_box = b
                self.assigned_iou = max(b.ious)
                self.type = 'positive'
            elif max(b.ious) > 0.70 and \
                    b == self.groundTruthBoxes[self.intersectionsOfUnions.index(max(self.intersectionsOfUnions))]:
                # if the anchor has an IoU > 0.70 with the ground truth box and if this box is from the anchors
                # point of view the box with the highest IoU then assign box to this anchor
                self.assigned_ground_truth_box = b
                self.type = 'positive'
        if max(self.intersectionsOfUnions) >= 0.3 and self.type != 'positive':
            self.type = 'neutral'
        elif self.type != 'positive':
            self.type = 'negative'
