import numpy as np
from PIL import Image
from math import ceil

from itertools import product as product

class PriorBox(object):
    def __init__(self, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        #output = torch.Tensor(anchors).view(-1, 4)
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), axis=1)
    return landms

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def detect_faces(loc, conf, landmarks, priors, img,conf_threshold=0.02):
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=np.float32)
    boxes = decode(loc[0], priors, [0.1, 0.2])
    resize =1
    boxes = boxes * scale / resize
    scores = conf[0][:, 1]
    tmp = [img.shape[1], img.shape[0], img.shape[1], img.shape[0], img.shape[1], img.shape[0], img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    scale1 = np.array(tmp, dtype=np.float32)

    landmarks = decode_landm(landmarks[0], priors, [0.1, 0.2])
    landmarks = landmarks * scale1 / resize

    # ignore low scores
    inds = np.where(scores > conf_threshold)[0]
    boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

    # sort
    order = scores.argsort()[::-1]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    # do NMS
    bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(bounding_boxes, 0.4)
    bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]
    return np.concatenate((bounding_boxes, landmarks), axis=1)

def crop_face(image, bboxes):
    try:
        b = bboxes
    except:
        print('huhu')

    landmarks_top = b[1]
    landmarks_bottom = b[3]
    landmarks_left = b[0]
    landmarks_right = b[2]
    top = int(landmarks_top - 0.35 * (landmarks_bottom - landmarks_top))
    bottom = int(landmarks_bottom + 0.2 * (landmarks_bottom - landmarks_top))
    left = int(landmarks_left - 0.15 * (landmarks_right - landmarks_left))
    right = int(landmarks_right + 0.15 * (landmarks_right - landmarks_left))
    if bottom - top > right - left:
        left -= ((bottom - top) - (right - left)) // 2
        right = left + (bottom - top)
    else:
        top -= ((right - left) - (bottom - top)) // 2
        bottom = top + (right - left)
    image_crop = np.ones(
        (bottom - top + 1, right - left + 1, 3), np.uint8) * 0
    h, w = image.shape[:2]
    left_white = max(0, -left)
    left = max(0, left)
    right = min(right, w-1)
    right_white = left_white + (right-left)
    top_white = max(0, -top)
    top = max(0, top)
    bottom = min(bottom, h-1)
    bottom_white = top_white + (bottom - top)
    image_crop[top_white:bottom_white+1, left_white:right_white +
                1] = image[top:bottom+1, left:right+1].copy()
    
    return Image.fromarray(image_crop)
