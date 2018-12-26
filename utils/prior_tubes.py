import numpy as np
import copy
import torch
from layers import act_cuboid_loss


class RGB_TUBES:
    def __init__(self, phase, use_gpu, variance=(0.1, 0.1, 0.2, 0.2), sequence_length=6):
        center_mode = phase == 'eval'
        self.conv4_3_norm_tubes = self.generate_prior_tubes(min_size=30.0, max_size=60.0, aspect_ratio=(2,), flip=True,
                                                            clip=False, layer_size=(38, 38), image_size=(300, 300),
                                                            step=8, offset=0.5,
                                                            sequence_length=sequence_length, center_mode=center_mode).reshape(-1, 4*sequence_length)
        self.fc_conv7_tubes = self.generate_prior_tubes(min_size=60.0, max_size=111.0, aspect_ratio=(2, 3,), flip=True,
                                                        clip=False, layer_size=(18, 18), image_size=(300, 300), step=16,
                                                        offset=0.5, sequence_length=sequence_length, center_mode=center_mode).reshape(-1, 4*sequence_length)
        self.conv6_tubes = self.generate_prior_tubes(min_size=111.0, max_size=162.0, aspect_ratio=(2, 3,), flip=True,
                                                     clip=False, layer_size=(9, 9), image_size=(300, 300), step=32,
                                                     offset=0.5, sequence_length=sequence_length, center_mode=center_mode).reshape(-1, 4*sequence_length)
        self.conv7_tubes = self.generate_prior_tubes(min_size=162.0, max_size=213.0, aspect_ratio=(2, 3,), flip=True,
                                                     clip=False, layer_size=(5, 5), image_size=(300, 300), step=64,
                                                     offset=0.5, sequence_length=sequence_length, center_mode=center_mode).reshape(-1, 4*sequence_length)
        self.conv8_tubes = self.generate_prior_tubes(min_size=213.0, max_size=264.0, aspect_ratio=(2,), flip=True,
                                                     clip=False, layer_size=(3, 3), image_size=(300, 300), step=100,
                                                     offset=0.5, sequence_length=sequence_length, center_mode=center_mode).reshape(-1, 4*sequence_length)
        self.conv9_tubes = self.generate_prior_tubes(min_size=264.0, max_size=315.0, aspect_ratio=(2,), flip=True,
                                                     clip=False, layer_size=(1, 1), image_size=(300, 300), step=300,
                                                     offset=0.5, sequence_length=sequence_length, center_mode=center_mode).reshape(-1, 4*sequence_length)
        if use_gpu:
            self.all_tubes = torch.from_numpy(np.vstack([self.conv4_3_norm_tubes, self.fc_conv7_tubes, self.conv6_tubes, self.conv7_tubes, self.conv8_tubes, self.conv9_tubes])).cuda()
            # self.all_tubes = torch.clamp(self.all_tubes, min=0, max=1)
        else:
            self.all_tubes = torch.from_numpy(np.vstack(
                [self.conv4_3_norm_tubes, self.fc_conv7_tubes, self.conv6_tubes, self.conv7_tubes, self.conv8_tubes,
                 self.conv9_tubes]))
            # self.all_tubes = torch.clamp(self.all_tubes, min=0, max=1)
        self.sequence_length = sequence_length
        self.variance = variance

    def generate_prior_tubes(self, min_size=30.0, max_size=None, aspect_ratio=(2,), flip=True, clip=False,
                             layer_size=(38, 38), image_size=(300, 300),
                             step=None, offset=0.5, sequence_length=6, center_mode=False):
        tubes_list = []
        if max_size is not None:
            num_priors = aspect_ratio.__len__() * 2 + 2
        else:
            num_priors = aspect_ratio.__len__() * 2 + 1
        if step is None:
            step_w = image_size[0] / layer_size[0]
            step_h = image_size[1] / layer_size[1]
        else:
            step_w = step
            step_h = step
        ar_list = []
        for a in aspect_ratio:
            ar_list.append(a)
            if flip:
                ar_list.append(1 / a)
        for h in range(layer_size[1]):
            for w in range(layer_size[0]):
                tube_set = np.zeros((num_priors, sequence_length, 4), dtype='float32')
                center_x, center_y = (w + offset) * step_w, (h + offset) * step_h
                box_width, box_height = min_size, min_size
                if center_mode:
                    tube_set[0, :, 0] = center_x / image_size[0]
                    tube_set[0, :, 1] = center_y / image_size[1]
                    tube_set[0, :, 2] = box_width / image_size[0]
                    tube_set[0, :, 3] = box_height / image_size[1]
                    if max_size is not None:
                        box_width, box_height = np.sqrt(min_size * max_size), np.sqrt(min_size * max_size)
                        tube_set[1, :, 0] = center_x / image_size[0]
                        tube_set[1, :, 1] = center_y / image_size[1]
                        tube_set[1, :, 2] = box_width / image_size[0]
                        tube_set[1, :, 3] = box_height / image_size[1]
                    prior_index = 2
                    for a in ar_list:
                        if (np.abs(a - 1.0) < 0.000001) or a < 0.000001:
                            continue
                        box_width, box_height = min_size * np.sqrt(a), min_size / np.sqrt(a)
                        tube_set[prior_index, :, 0] = center_x / image_size[0]
                        tube_set[prior_index, :, 1] = center_y / image_size[1]
                        tube_set[prior_index, :, 2] = box_width / image_size[0]
                        tube_set[prior_index, :, 3] = box_height / image_size[1]
                        prior_index += 1
                else:
                    tube_set[0, :, 0] = (center_x - box_width/2.0) / image_size[0]  # xmin
                    tube_set[0, :, 1] = (center_y - box_height/2.0) / image_size[1]  # ymin
                    tube_set[0, :, 2] = (center_x + box_width/2.0) / image_size[0]
                    tube_set[0, :, 3] = (center_y + box_height/2.0) / image_size[1]  # ymax
                    if max_size is not None:
                        box_width, box_height = np.sqrt(min_size * max_size), np.sqrt(min_size * max_size)
                        tube_set[1, :, 0] = (center_x - box_width / 2.0) / image_size[0]  # xmin
                        tube_set[1, :, 1] = (center_y - box_height / 2.0) / image_size[1]  # ymin
                        tube_set[1, :, 2] = (center_x + box_width / 2.0) / image_size[0]
                        tube_set[1, :, 3] = (center_y + box_height / 2.0) / image_size[1]  # ymax
                    prior_index = 2
                    for a in ar_list:
                        if (np.abs(a - 1.0) < 0.000001) or a < 0.000001:
                            continue
                        box_width, box_height = min_size * np.sqrt(a), min_size / np.sqrt(a)
                        tube_set[prior_index, :, 0] = (center_x - box_width / 2.0) / image_size[0]  # xmin
                        tube_set[prior_index, :, 1] = (center_y - box_height / 2.0) / image_size[1]  # ymin
                        tube_set[prior_index, :, 2] = (center_x + box_width / 2.0) / image_size[0]
                        tube_set[prior_index, :, 3] = (center_y + box_height / 2.0) / image_size[1]  # ymax
                        prior_index += 1
                if clip:
                    tube_set[tube_set > 1.0] = 1.0
                    tube_set[tube_set < 0.0] = 0.0
                tubes_list.append(tube_set)
        return np.vstack(tubes_list)  # 这里得到的结果是按照顺序的各个tubes，和feature map上的顺序是完全对应的


def get_all_video_tubes(tubes):
    return copy.deepcopy(tubes.all_tubes)


def decode_tubes(tubes, loc_preds=None):  # just for one video
    decode_video_tubes = get_all_video_tubes(tubes)
    decode_video_tubes = torch.stack([decode_video_tubes for i in range(loc_preds.shape[0])], dim=0)
    var = tubes.variance
    center_x = decode_video_tubes[:, :, 0::4]
    center_y = decode_video_tubes[:, :, 1::4]
    width = decode_video_tubes[:, :, 2::4]
    height = decode_video_tubes[:, :, 3::4]
    new_center_x = var[0] * loc_preds[:, :, 0::4] * width + center_x
    new_center_y = var[1] * loc_preds[:, :, 1::4] * height + center_y
    new_width = torch.exp(var[2] * loc_preds[:, :, 2::4]) * width
    new_height = torch.exp(var[3] * loc_preds[:, :, 3::4]) * height
    decode_video_tubes[:, :, 0::4] = new_center_x - new_width / 2.0  # x_min
    decode_video_tubes[:, :, 1::4] = new_center_y - new_height / 2.0  # y_min
    decode_video_tubes[:, :, 2::4] = new_center_x + new_width / 2.0  # x_max
    decode_video_tubes[:, :, 3::4] = new_center_y + new_height / 2.0  # y_max
    decode_video_tubes[:, :, 0::4] = torch.clamp(decode_video_tubes[:, :, 0::4], min=0)
    decode_video_tubes[:, :, 1::4] = torch.clamp(decode_video_tubes[:, :, 1::4], min=0)
    decode_video_tubes[:, :, 2::4] = torch.clamp(decode_video_tubes[:, :, 2::4], max=1)
    decode_video_tubes[:, :, 3::4] = torch.clamp(decode_video_tubes[:, :, 3::4], max=1)
    return decode_video_tubes


def get_tubes_conf(conf_preds_list=None, num_class=25):
    # 这个函数按照tubes的顺序把所有的confidence 打分提取出来
    conf_list = []
    for conf_preds in conf_preds_list:
        batch_num, channel_num, w, h = conf_preds.shape
        feature_flat = conf_preds.detach().numpy().reshape((channel_num, w * h))
        prior_num = int(channel_num / num_class)
        for i in range(w * h):
            for j in range(prior_num):
                conf = feature_flat[j * num_class:(j + 1) * num_class, i].reshape(1, -1)
                conf_list.append(conf)
    return np.vstack(conf_list)


def apply_nms(tubes_conf, decode_tubes, conf_threshold=0.01, nms_threshold=0.45, nms_top_k=400, keep_topk=200,
              num_class=25):
    nms_tubes_list = []
    nms_scores_list = []
    nms_label_list = []
    for i in range(1, num_class):  # 不要背景的
        scores_c = tubes_conf[:, i]
        select = scores_c > conf_threshold
        if select.sum() > 0:
            select_tubes = decode_tubes[select, :]
            scores_c = scores_c[select]
            sort_index = torch.argsort(-scores_c)
            if sort_index.shape[0] > nms_top_k:
                sort_index = sort_index[:nms_top_k]
            class_nms_index_list = [sort_index[0]]
            class_nms_tube = select_tubes[sort_index[0], :].unsqueeze(dim=0)
            for index in sort_index[1:]:
                ioutable = torch.zeros(class_nms_index_list.__len__())
                act_cuboid_loss.get_tube_overlap(class_nms_tube, select_tubes[index, :], ioutable)
                if (ioutable > nms_threshold).sum() == 0:
                    class_nms_index_list += [index]
                    class_nms_tube = torch.cat([class_nms_tube, select_tubes[index, :].unsqueeze(dim=0)], dim=0)
            for k, index in enumerate(class_nms_index_list):
                nms_tubes_list += [select_tubes[index, :]]
                nms_scores_list += [scores_c[index]]
                nms_label_list += [i]
    return_tubes_list = []
    return_scores_list = []
    return_label_list = []
    nms_scores = torch.Tensor(nms_scores_list)
    nms_scores_index = torch.argsort(-nms_scores)
    if nms_tubes_list.__len__() > keep_topk:
        for index in nms_scores_index[:keep_topk]:
            return_tubes_list += [nms_tubes_list[index]]
            return_scores_list += [nms_scores[index]]
            return_label_list += [nms_label_list[index]]
    else:
        for index in nms_scores_index:
            return_tubes_list += [nms_tubes_list[index]]
            return_scores_list += [nms_scores[index]]
            return_label_list += [nms_label_list[index]]
    return return_tubes_list, return_scores_list, return_label_list