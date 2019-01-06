import numpy as np
import pickle
import os
from utils import map_eval
from utils import box
import cv2
import matplotlib.pyplot as plt


def error_analyse(dataset):
    error_images_path = "/mnt/data/qzw/result/pytorch-act-detector/{}/error_images/".format(dataset.DNAME)
    result_file = "/mnt/data/qzw/result/pytorch-act-detector/{}/all_frame_boxes_list-{}-{}-0.8601.pkl".format(dataset.DNAME, dataset.DNAME, dataset.modality)
    if not os.path.isfile(result_file):
        raise ValueError("file:{}   not found".format(result_file))
    with open(result_file, "rb") as file:
        all_frame_boxes_list = pickle.load(file)
    labels = dataset.get_labels()
    gt_tubes = dataset.get_gttubes()
    gt_dict, gt_label_num = map_eval.get_ground_truth(dataset.videos_list, labels, gt_tubes)
    all_frame_boxes = np.vstack(all_frame_boxes_list)
    frame_gt_box_dict = {}
    frame_error_box_dict = {}
    frame_correct_box_dict = {}
    label_pr_dict = {}
    for label in range(labels.__len__()):
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
        video_name_list = []
        for cnt, id in enumerate(pre_idx):
            pre_box = label_pre_box[id, :]
            video_name_list += [int(pre_box[0])-1]
            # video_label = list(dataset.gttubes[dataset.videos_list[int(pre_box[0])-1]].keys())[0]
            # if labels[label] == 'Run' and dataset.labels[video_label] == 'SkateBoarding':
            #     continue
            positive = False
            if (int(pre_box[0]), int(pre_box[1]), int(pre_box[2])) in gt_dict:
                _gt = gt_dict[(int(pre_box[0]), int(pre_box[1]), int(pre_box[2]))]

                if (int(pre_box[0]), int(pre_box[1])) not in frame_gt_box_dict:
                    frame_gt_box_dict[(int(pre_box[0]), int(pre_box[1]))] = []
                frame_gt_box_dict[(int(pre_box[0]), int(pre_box[1]))] += _gt.copy()

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
                if (int(pre_box[0]), int(pre_box[1])) not in frame_correct_box_dict:
                    frame_correct_box_dict[(int(pre_box[0]), int(pre_box[1]))] = []
                frame_correct_box_dict[(int(pre_box[0]), int(pre_box[1]))] += [pre_box]
            else:
                fp += 1
                if (int(pre_box[0]), int(pre_box[1])) not in frame_error_box_dict:
                    frame_error_box_dict[(int(pre_box[0]), int(pre_box[1]))] = []
                frame_error_box_dict[(int(pre_box[0]), int(pre_box[1]))] += [pre_box]
                # video_name = dataset.videos_list[int(pre_box[0])-1]
                # video_label = list(dataset.gttubes[video_name].keys())[0]
                # err_image_root = os.path.join(error_images_path, video_name+"-"+dataset.labels[video_label])
                # if os.path.exists(err_image_root) is not True:
                #     os.mkdir(err_image_root)
                # image = cv2.imread(os.path.join(dataset.data_path, 'Frames', video_name, dataset.image_format % int(pre_box[1])))
                # draw_rec_and_save_image(_gt, pre_box, dataset.labels[int(pre_box[2])], image, err_image_root)
            pr[pr_cnt, 0] = tp / (fp + tp)
            pr[pr_cnt, 1] = tp / (tp + fn)
            if labels[label] == 'SkateBoarding' and cnt < 1000:
                pause = 0
                image = cv2.imread(os.path.join(dataset.data_path, 'Frames', dataset.videos_list[int(pre_box[0]) - 1],
                                                dataset.image_format % int(pre_box[1])))
                err_image_root = os.path.join(error_images_path, labels[label])
                if os.path.exists(err_image_root) is not True:
                    os.mkdir(err_image_root)
                if positive:
                    draw_rec_and_save_image([], [pre_box], [], labels, image, err_image_root, cnt)
                else:
                    draw_rec_and_save_image([], [], [pre_box], labels, image, err_image_root, cnt)
            pr_cnt += 1
        video_name_list = np.array(video_name_list).reshape(-1, 1)
        label_pr_dict[label] = pr
        # plt.cla()
        # plt.plot(pr[:, 1], pr[:, 0], color='blue')
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.savefig('./{}.jpg'.format(labels[label]))
    # exit(0)
    # for i, video in enumerate(dataset.videos_list):
    #     video_label = list(dataset.gttubes[video].keys())[0]
    #     if dataset.labels[video_label] != 'Walk':
    #         continue
    #     nframes = os.listdir(os.path.join(dataset.data_path, 'Frames', video)).__len__()
    #     print("video index:", i)
        # for j in range(nframes):
        #     image = cv2.imread(os.path.join(dataset.data_path, 'Frames', video, dataset.image_format % int(j+1)))
        #     video_label = list(dataset.gttubes[video].keys())[0]
        #     err_image_root = os.path.join(error_images_path, video + "-" + dataset.labels[video_label])
        #     if os.path.exists(err_image_root) is not True:
        #         os.mkdir(err_image_root)
        #     if (i+1, j+1) in frame_gt_box_dict:
        #         gt = frame_gt_box_dict[(i+1, j+1)]
        #     else:
        #         gt = []
        #     if (i+1, j+1) in frame_correct_box_dict:
        #         cpb = frame_correct_box_dict[(i+1, j+1)]
        #     else:
        #         cpb = []
        #     if (i+1, j+1) in frame_error_box_dict:
        #         epb = frame_error_box_dict[(i+1, j+1)]
        #     else:
        #         epb = []
        #     draw_rec_and_save_image(gt, cpb, epb, labels, image, err_image_root, j+1)

    ap = np.empty(labels.__len__())
    for label in label_pr_dict:
        prdif = label_pr_dict[label][1:, 1] - label_pr_dict[label][:-1, 1]
        prsum = label_pr_dict[label][1:, 0] + label_pr_dict[label][:-1, 0]
        ap[label] = np.sum(prdif * prsum * 0.5)
    mmap = np.mean(ap)
    print("map:", mmap)
    return mmap


def draw_rec_and_save_image(ground_truths, correct_pre_boxes, error_pre_boxes, labels, image, image_save_path, frame_index):
    for gt in ground_truths:
        p1 = (int(gt[0]), int(gt[1]))
        p2 = (int(gt[2]), int(gt[3]))
        cv2.rectangle(image, p1, p2, (0, 255, 0))
    for pb in error_pre_boxes:
        if pb[3] < 0.1:
            continue
        p1 = (int(pb[4]), int(pb[5]))
        p2 = (int(pb[6]), int(pb[7]))
        pt = (int(pb[4]), int(pb[7]))
        cv2.rectangle(image, p1, p2, (0, 0, 255))
        cv2.putText(image, "conf:%.3f  " % pb[3] + labels[int(pb[2])], pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (32, 32, 32))
    for pb in correct_pre_boxes:
        p1 = (int(pb[4]), int(pb[5]))
        p2 = (int(pb[6]), int(pb[7]))
        pt = (int(pb[4]), int(pb[5]+10))
        cv2.rectangle(image, p1, p2, (255, 0, 0))
        cv2.putText(image, "conf:%.3f  " % pb[3] + labels[int(pb[2])], pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (132, 132, 32))
    cv2.imwrite(os.path.join(image_save_path, "frame-{}.jpg").format(int(frame_index)), image)


if __name__ == '__main__':
    # from data import dataset
    # dataset_name = 'UCFSports'
    # modality = 'rgb'
    # data_path = "/mnt/data/qzw/data/{}/".format(dataset_name)
    # analyse_data_set = dataset.TubeDataset(dataset_name, data_path=data_path, phase='eval',
    #                                        modality=modality,
    #                                        sequence_length=6)
    # error_analyse(analyse_data_set)
    import config
    from data import dataset
    args = config.Config()
    with open('./error_random_crop_data.pkl', 'rb') as f:
        index = pickle.load(f)
        image_list = pickle.load(f)
        ground_truth = pickle.load(f)
        _random_crop_data = pickle.load(f)
        # _random_flip_data = pickle.load(f)
    train_dataset = dataset.TubeDataset(args.dataset, data_path=args.data_path, phase='train',
                                modality=args.modality,
                                sequence_length=6)
    a, b = train_dataset[index]


