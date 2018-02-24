class GroundTruthBox:

    def __init__(self, x, y, w, h, label):
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
        self.anchors.append(anchor)
        self.ious.append(iou)

    def get_best_anchor(self):
        if len(self.anchors) > 0:
            index = self.ious.index(max(self.ious))
            return self.anchors[index], self.ious[index]
        return None
