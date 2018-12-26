import numpy as np

def jaccard_overlap_boxes(box1, box2):
    # box.shape=[xmin,ymin,xmax,ymax]
    if box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1]:
        return 0.0
    else:
        box = np.array([max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])])
        size = box_size(box)
        if box_size(box1) + box_size(box2) < size:
            size = size
        return size / (box_size(box1) + box_size(box2) - size)


def box_size(box):
    # box.shape=[xmin,ymin,xmax,ymax]
    return (box[2] - box[0]) * (box[3] - box[1])
