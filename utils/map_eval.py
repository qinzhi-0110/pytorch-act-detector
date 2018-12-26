import pickle
import numpy as np
from . import box


def nms_class(boxes_scores_list, nms_threshold):
    if boxes_scores_list.__len__() <= 1:
        return [0]
    boxes = np.vstack(boxes_scores_list)
    scores_index = np.argsort(-boxes[:, 0])
    class_nms_index_list = [scores_index[0]]
    for index in scores_index[1:]:
        keep = True
        for _max_index in class_nms_index_list:
            if box.jaccard_overlap_boxes(boxes[index, 1:], boxes[_max_index, 1:]) > nms_threshold:
                keep = False
                break
        if keep:
            class_nms_index_list += [index]
    return class_nms_index_list


def get_pr(data_cache, K=6):
    tubelet_file = './data/UCFSports/{}/{}.pkl'
    output_file = './data/UCFSports_build_tubes/pr_data.pkl'
    test_videos_list = data_cache.get_test_videos()
    nframes_dict = data_cache.get_nframes()
    labels = data_cache.get_labels()
    gt_tubes = data_cache.get_gttubes()
    resolution = data_cache._resolution
    gt_dict = {}
    gt_label_num = np.zeros(labels.__len__())
    video_index = 0
    for video in test_videos_list[0]:
        video_index += 1
        for label in gt_tubes[video]:
            for tube in gt_tubes[video][label]:
                for i in range(tube.shape[0]):
                    if (video_index, int(tube[i, 0]), label) not in gt_dict:
                        gt_dict[(video_index, int(tube[i, 0]), label)] = []
                    gt_dict[(video_index, int(tube[i, 0]), label)] += [tube[i, 1:]]
                    gt_label_num[label] += 1

    frame_boxes = {}
    all_frame_boxes_list = []
    video_index = 0
    for videos in test_videos_list[0]:
        nframes = nframes_dict[videos]
        frame_boxes[videos] = {}
        video_index += 1
        height, width = resolution[videos]
        for start_frame in range(1, nframes - K + 2):
            file = open(tubelet_file.format(videos, start_frame), 'rb')
            _, __, best_tube = pickle.load(file)
            file.close()
            for i in range(best_tube.shape[0]):
                for j in range(K):
                    if (j+start_frame) not in frame_boxes[videos]:
                        frame_boxes[videos][j+start_frame] = []
                    frame_boxes[videos][j+start_frame] += [best_tube[i, np.array([0, 1, 2+4*j, 3+4*j, 4+4*j, 5+4*j])]]

        for frame_index in range(1, nframes+1):
            frame_label = {}
            for bb in frame_boxes[videos][frame_index]:
                if bb[0] not in frame_label:
                    frame_label[bb[0]] = []
                frame_label[bb[0]] += [bb[1:]]

            for tt in frame_label:
                idx = nms_class(frame_label[tt], nms_threshold=0.3)
                for id in idx:
                    all_frame_boxes_list += [np.hstack([np.array([video_index, frame_index, tt]), frame_label[tt][id] * np.array([1, width, height, width, height])])]

    all_frame_boxes = np.vstack(all_frame_boxes_list)
    label_pr_dict = {}
    for label in range(labels.__len__()):
        print("label:", label)
        pre_idx = np.where(all_frame_boxes[:, 2] == label)[0]
        label_pre_box = all_frame_boxes[pre_idx]
        pre_idx = np.argsort(-label_pre_box[:, 3])
        pr = np.empty((pre_idx.shape[0]+1, 2))
        pr[0, 0] = 1.0 #  precision
        pr[0, 1] = 0.0 #  recall
        pr_cnt = 1
        fn = gt_label_num[label]
        fp = 0
        tp = 0
        for id in pre_idx:
            pre_box = label_pre_box[id, :]
            positive = False
            if (int(pre_box[0]), int(pre_box[1]), int(pre_box[2])) in gt_dict:
                _gt = gt_dict[(int(pre_box[0]), int(pre_box[1]), int(pre_box[2]))]
                ious = np.zeros(_gt.__len__())
                for i, g in enumerate(_gt):
                    ious[i] = box.jaccard_overlap_boxes(pre_box[4:], g)
                i_max = np.argmax(ious)
                if ious[i_max] > 0.5:
                    positive = True
                    del _gt[i_max]
                    if _gt.__len__() == 0:
                        del gt_dict[(int(pre_box[0]), int(pre_box[1]), int(pre_box[2]))]
            if positive:
                tp += 1
                fn -= 1
            else:
                fp += 1
            pr[pr_cnt, 0] = tp / (fp + tp)
            pr[pr_cnt, 1] = tp / (tp + fn)
            pr_cnt += 1
        label_pr_dict[label] = pr
    with open(output_file, 'wb') as f:
        pickle.dump(label_pr_dict, f)
    ap = np.empty(labels.__len__())
    for label in label_pr_dict:
        prdif = label_pr_dict[label][1:, 1] - label_pr_dict[label][:-1, 1]
        prsum = label_pr_dict[label][1:, 0] + label_pr_dict[label][:-1, 0]
        ap[label] = np.sum(prdif * prsum * 0.5)
    print("map:", np.mean(ap))


def get_ground_truth(test_videos_list, labels, gt_tubes):
    gt_dict = {}
    gt_label_num = np.zeros(labels.__len__())
    video_index = 0
    for video in test_videos_list:
        video_index += 1
        for label in gt_tubes[video]:
            for tube in gt_tubes[video][label]:
                for i in range(tube.shape[0]):
                    if (video_index, int(tube[i, 0]), label) not in gt_dict:
                        gt_dict[(video_index, int(tube[i, 0]), label)] = []
                    gt_dict[(video_index, int(tube[i, 0]), label)] += [tube[i, 1:]]
                    gt_label_num[label] += 1
    return gt_dict, gt_label_num


def calc_pr(all_frame_boxes_list, dataset):
    output_file = './pr_data_{}_{}.pkl'.format(dataset.DNAME, dataset.modality)
    labels = dataset.get_labels()
    gt_tubes = dataset.get_gttubes()
    gt_dict, gt_label_num = get_ground_truth(dataset.videos_list, labels, gt_tubes)
    all_frame_boxes = np.vstack(all_frame_boxes_list)
    label_pr_dict = {}
    for label in range(labels.__len__()):
        # print("label:", label)
        pre_idx = np.where(all_frame_boxes[:, 2] == label)[0]
        label_pre_box = all_frame_boxes[pre_idx]
        pre_idx = np.argsort(-label_pre_box[:, 3])
        pr = np.empty((pre_idx.shape[0]+1, 2))
        pr[0, 0] = 1.0 #  precision
        pr[0, 1] = 0.0 #  recall
        pr_cnt = 1
        fn = gt_label_num[label]
        fp = 0
        tp = 0
        for id in pre_idx:
            pre_box = label_pre_box[id, :]
            positive = False
            if (int(pre_box[0]), int(pre_box[1]), int(pre_box[2])) in gt_dict:
                _gt = gt_dict[(int(pre_box[0]), int(pre_box[1]), int(pre_box[2]))]
                ious = np.zeros(_gt.__len__())
                for i, g in enumerate(_gt):
                    ious[i] = box.jaccard_overlap_boxes(pre_box[4:], g)
                i_max = np.argmax(ious)
                if ious[i_max] > 0.5:
                    positive = True
                    del _gt[i_max]
                    if _gt.__len__() == 0:
                        del gt_dict[(int(pre_box[0]), int(pre_box[1]), int(pre_box[2]))]
            if positive:
                tp += 1
                fn -= 1
            else:
                fp += 1
            pr[pr_cnt, 0] = tp / (fp + tp)
            pr[pr_cnt, 1] = tp / (tp + fn)
            pr_cnt += 1
        label_pr_dict[label] = pr

    with open(output_file, 'wb') as f:
        pickle.dump(label_pr_dict, f)
    ap = np.empty(labels.__len__())
    for label in label_pr_dict:
        prdif = label_pr_dict[label][1:, 1] - label_pr_dict[label][:-1, 1]
        prsum = label_pr_dict[label][1:, 0] + label_pr_dict[label][:-1, 0]
        ap[label] = np.sum(prdif * prsum * 0.5)
    mmap = np.mean(ap)
    print("map:", mmap)
    return mmap


if __name__ == '__main__':
    data_cache = tube_dataset.TubeDataset('UCFSports')
    get_pr(data_cache, K=6)

