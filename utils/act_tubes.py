import numpy as np
import pickle
import tube_dataset
import ACT_utils
import box
import prior_tubes


def nms_tublets(tubes_conf, decode_tubes, nms_threshold=0.45, top_k=400):
    cnt = 0
    # tubes_conf = tubes_conf.squeeze().detach().numpy()
    # decode_tubes = decode_tubes.reshape(-1, 6, 4)
    # max_class = np.max(tubes_conf[:, 1:], axis=1)
    class_index = np.argsort(-tubes_conf)
    class_nms_index_list = [class_index[0]]
    for index in class_index[1:]:
        keep = True
        for _max_index in class_nms_index_list:
            if prior_tubes.jaccard_overlap_tubes(decode_tubes[index, :], decode_tubes[_max_index, :]) > nms_threshold:
                keep = False
                break
        if keep:
            class_nms_index_list += [index]
            cnt += 1
            if cnt >= top_k:
                break
    return np.array(class_nms_index_list)[:top_k]


def nms_tublets_caffe(tubes_conf, decode_tubes, nms_threshold=0.45, top_k=400, K=6):
    counter = 0
    x1 = [decode_tubes[:, i, 0] for i in range(K)]
    y1 = [decode_tubes[:, i, 1] for i in range(K)]
    x2 = [decode_tubes[:, i, 2] for i in range(K)]
    y2 = [decode_tubes[:, i, 3] for i in range(K)]
    dets = tubes_conf
    area = [(x2[k] - x1[k]) * (y2[k] - y1[k]) for k in range(K)]
    I = np.argsort(dets)
    indices = np.empty(top_k, dtype=np.int32)

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1

        # Compute overlap
        xx1 = [np.maximum(x1[k][i], x1[k][I[:-1]]) for k in range(K)]
        yy1 = [np.maximum(y1[k][i], y1[k][I[:-1]]) for k in range(K)]
        xx2 = [np.minimum(x2[k][i], x2[k][I[:-1]]) for k in range(K)]
        yy2 = [np.minimum(y2[k][i], y2[k][I[:-1]]) for k in range(K)]

        w = [np.maximum(0, xx2[k] - xx1[k]) for k in range(K)]
        h = [np.maximum(0, yy2[k] - yy1[k]) for k in range(K)]

        inter_area = [w[k] * h[k] for k in range(K)]
        ious = sum([inter_area[k] / (area[k][I[:-1]] + area[k][i] - inter_area[k]) for k in range(K)])

        I = I[np.where(ious <= nms_threshold * K)[0]]

        if counter == top_k: break

    return indices[:counter]


if __name__ == '__main__':
    data_cache = tube_dataset.TubeDataset('UCFSports')
    build_tubes(data_cache, K=6)

