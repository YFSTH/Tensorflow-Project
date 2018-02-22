class Anchor(object):

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.groundTruthBoxes = []
        self.intersectionsOfUnions = []
        self.mnist_classes = []
        self.type = 'negative'

    def append_ground_truth_box(self, ground_truth_box):
        iou = self.calculate_iou(ground_truth_box)
        if iou > 0:
            self.intersectionsOfUnions.append(iou)
            self.groundTruthBoxes.append(ground_truth_box)
            self.append_mnist_class()
        self.evaluate_type()

    def calculate_iou(self, ground_truth_box):
        '''
        Calculates the intersection over union of anchor and ground truth box
        :param ground_truth_box: tuple with (class, x, y, w, h, angle, scale)
        :return: intersection over union
        '''
        # calculate most upper-left (x1,y1) and bottom-right (x2,y2) pixel coordinates of
        # anchor and ground truth box
        x1_a = self.x - self.w
        x2_a = self.x + self.w
        y1_a = self.y - self.h
        y2_a = self.y + self.h
        _, x_t, y_t, w_t, h_t, _, _ = ground_truth_box
        x1_t = x_t - w_t
        x2_t = x_t + w_t
        y1_t = y_t - h_t
        y2_t = y_t + h_t

        # calculate coordinates of intersection rectangle
        x1_i = max(x1_a, x1_t)
        y1_i = max(y1_a, y1_t)
        x2_i = max(x2_a, x2_t)
        y2_i = max(y2_a, y2_t)

        # areas of anchor and ground truth box and intersection
        area_a = self.w * self.h
        area_t = w_t * h_t
        intersection = (x2_i - x1_i) * (y2_i - y1_i + 1)

        # calculate and return intersection over union
        return intersection / (area_a + area_t - intersection)

    def append_mnist_class(self, ground_truth_box):
        label = ground_truth_box[0]
        self.mnist_classes.append(label)

    def evaluate_type(self):
        if max(self.intersectionsOfUnions) >= 0.7:
            self.type = 'positive'
        elif max(self.intersectionsOfUnions) >= 0.3:
            self.type = 'neutral'
        else:
            self.type = 'negative'


    # TODO: Implement Ground truth boxes class --> because Anchor can also be positive if it is the only anchor
    # TODO: ... intersecting with the respective ground truth box



    def get_ground_truth_box_with_highest_iou(self):
        '''
        :return: ground truth box with highest IoU, IoU, label of ground truth box, if no ground truth
                 box was appended
        '''
        if len(self.groundTruthBoxes) > 0:
            max_index = self.intersectionsOfUnions.index(max(self.intersectionsOfUnions))
            return self.groundTruthBoxes[max_index], self.intersectionsOfUnions[max_index], self.labels[max_index]
        return None

