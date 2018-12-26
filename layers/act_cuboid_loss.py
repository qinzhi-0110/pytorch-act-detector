import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import prior_tubes


def get_tube_overlap(tube1, tube2, ioutable):
    ground_truth = tube2.expand(tube1.shape)
    total_tube = torch.cat([tube1.unsqueeze(0), ground_truth.unsqueeze(0)], dim=0)

    xmin = torch.max(total_tube[:, :, 0::4], dim=0)[0]
    ymin = torch.max(total_tube[:, :, 1::4], dim=0)[0]
    xmax = torch.min(total_tube[:, :, 2::4], dim=0)[0]
    ymax = torch.min(total_tube[:, :, 3::4], dim=0)[0]

    cross_area = torch.clamp(xmax - xmin, min=0)*torch.clamp(ymax - ymin, min=0)

    valid_area = cross_area.sum(dim=1) > 0
    valid = valid_area.unsqueeze(1).expand(tube1.shape)
    valid_priortubes = tube1[valid].view((-1, tube1.shape[1]))

    prior_area = (valid_priortubes[:, 2::4] - valid_priortubes[:, 0::4])*(valid_priortubes[:, 3::4] - valid_priortubes[:, 1::4])
    valid = valid_area.unsqueeze(1).expand(-1, prior_area.shape[1])
    valid_cross_area = cross_area[valid].view((-1, prior_area.shape[1]))

    gt_area = (tube2[2::4] - tube2[0::4])*(tube2[3::4] - tube2[1::4])

    ratio = valid_cross_area/(gt_area + prior_area - valid_cross_area)
    ratio = ratio.sum(dim=1)
    ioutable[valid_area] = ratio/prior_area.shape[1]


class CuboidLoss(nn.Module):
    def __init__(self, use_gpu, variance, num_class, k_frames):
        super(CuboidLoss, self).__init__()
        self.use_gpu = use_gpu
        self.variance = variance
        self.k_frames = k_frames
        self.num_class = num_class
        self.tubes_init = prior_tubes.RGB_TUBES(phase='train', use_gpu=use_gpu)


    def ACTMatchTube(self, prior_tubes, ground_truth):
        # prior_tubes.shape = (8396*24), it's same to all tubes
        # ground_truth is a tensor ,,every batch just one ground truth,,ground_truth.shape = (bath_num*sequence*(1+4))
        # if self.use_gpu:
        #     _ground_truth = ground_truth.cpu().numpy()
        # else:
        #     _ground_truth = ground_truth.numpy()
        batch_num = ground_truth.shape[0]
        prior_tubes_num = prior_tubes.shape[0]
        tubes_label = torch.zeros((batch_num, prior_tubes_num, self.num_class), dtype=torch.uint8)
        tubes_label_index = torch.zeros((batch_num, prior_tubes_num), dtype=torch.int64)
        tubes_label[:, :, 0] = 1
        positive_samples_index_list = []
        N = 0
        for i in range(batch_num):
            if self.use_gpu:
                iou_table = torch.zeros(prior_tubes_num, dtype=torch.float32).cuda()
            else:
                iou_table = torch.zeros(prior_tubes_num, dtype=torch.float32)
            # for prior in range(prior_tubes_num):
            #     iou_table[prior] = get_tube_overlap(prior_tubes[prior, :], _ground_truth[i, 1:], self.k_frames)
            get_tube_overlap(prior_tubes, ground_truth[i, 1:], iou_table)
            positive_sample_index = []
            max_prior_index = torch.argmax(iou_table, 0)
            positive_sample_index += [(max_prior_index, i)]
            tubes_label_index[i, max_prior_index] = int(ground_truth[i, 0])
            tubes_label[i, max_prior_index, int(ground_truth[i, 0])] = 1
            tubes_label[i, max_prior_index, 0] = 0
            pp = torch.argsort(-iou_table)
            for tt in pp:
                if iou_table[tt] < 0.5:
                    break
                if tubes_label[i, tt, 0] == 1:
                    positive_sample_index += [(tt, i)]
                    tubes_label_index[i, tt] = int(ground_truth[i, 0])
                    tubes_label[i, tt, int(ground_truth[i, 0])] = 1
                    tubes_label[i, tt, 0] = 0
            N += positive_sample_index.__len__()
            positive_samples_index_list += [torch.tensor(positive_sample_index)]
        return positive_samples_index_list, tubes_label, tubes_label_index, N

    def ACTComputeConfLoss(self, conf_preds, tubes_label):
        conf_preds_max = torch.max(conf_preds, dim=-1)[0].unsqueeze(-1)
        my_conf_preds = F.softmax(conf_preds - conf_preds_max, dim=-1)
        aa = my_conf_preds[tubes_label]
        aa = aa.view(tubes_label.shape[0], tubes_label.shape[1])
        tubes_loss = -torch.log(aa + 0.000001)
        return tubes_loss

    def ACTMineHardExamples(self, tubes_loss, positive_samples_index_list):
        negtive_samples_index_list = []
        if self.use_gpu:
            tubes_loss = tubes_loss.cpu().detach().numpy()
        else:
            tubes_loss = tubes_loss.detach().numpy()
        for i in range(tubes_loss.shape[0]):
            positive_sample_index = positive_samples_index_list[i]
            positive_num = positive_sample_index.shape[0]
            negtive_num = 3 * positive_num
            hard_examples_index = []
            tube_loss = tubes_loss[i, :]
            max_index = np.argsort(-tube_loss)
            negtive_count = 0
            for index in max_index:
                if index not in positive_sample_index[:, 0]:
                    hard_examples_index += [index]
                    negtive_count += 1
                    if negtive_count >= negtive_num:
                        break
            negtive_samples_index_list += [np.array(hard_examples_index)]
        return negtive_samples_index_list

    def ACTGetLocLoss(self, loc_preds, positive_samples_index_list, prior_tubes, ground_truth):
        # ground_truth is a list ,,its len is batch_num, its element.shape = gt_num*(1+24), the first is label
        batch_num = loc_preds.shape[0]
        if self.use_gpu:
        #     _prior_tubes = torch.from_numpy(prior_tubes).cuda()
            encode_loc = torch.zeros(loc_preds.shape).cuda()
            pos_index = torch.zeros(loc_preds.shape, dtype=torch.uint8).cuda()
        else:
        #     _prior_tubes = torch.from_numpy(prior_tubes)
            encode_loc = torch.zeros(loc_preds.shape)
            pos_index = torch.zeros(loc_preds.shape, dtype=torch.uint8)
        for i in range(batch_num):
            positive_samples_index = positive_samples_index_list[i]
            for j in range(positive_samples_index.shape[0]):
                pos_index[i, positive_samples_index[j, 0], :] = 1
                self.EncodeTube(prior_tubes[positive_samples_index[j, 0], :], ground_truth[i, 1:],
                                encode_loc[i, positive_samples_index[j, 0], :])
        encode_loc = encode_loc.requires_grad = False
        loc_p = loc_preds[pos_index].view(-1, self.k_frames * 4)
        loc_t = encode_loc[pos_index].view(-1, self.k_frames * 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) / self.k_frames
        return loss_l

    def ACTGetConfLoss(self, conf_preds, positive_samples_index_list, negtive_samples_index_list, tubes_label):
        '''
        :param conf_preds:
        :param positive_samples_index_list:
        :param negtive_samples_index_list:
        :param tubes_label: (batch_num * 8396)
        :return:
        '''
        batch_num = conf_preds.shape[0]
        prior_num = conf_preds.shape[1]
        conf_pos_index = torch.zeros(conf_preds.shape, dtype=torch.uint8)
        target_pos_index = torch.zeros((batch_num, prior_num), dtype=torch.uint8)
        for i in range(batch_num):
            positive_samples_index = positive_samples_index_list[i]
            negtive_samples_index = negtive_samples_index_list[i]
            for j in range(positive_samples_index.shape[0]):
                conf_pos_index[i, positive_samples_index[j, 0], :] = 1
                target_pos_index[i, positive_samples_index[j, 0]] = 1
            for j in range(negtive_samples_index.shape[0]):
                conf_pos_index[i, negtive_samples_index[j], :] = 1
                target_pos_index[i, negtive_samples_index[j]] = 1
        tubes_label = Variable(tubes_label, requires_grad=False)
        if self.use_gpu:
            tubes_label = tubes_label.cuda()
        conf_p = conf_preds[conf_pos_index].view(-1, self.num_class)
        target_weights = tubes_label[target_pos_index]
        loss_c = F.cross_entropy(conf_p, target_weights, size_average=True)
        return loss_c

    def EncodeTube(self, prior_tube, gt_tube, encode):
        '''
        prior_tube=(xmin, ymin, xmax, ymax)*sequence_length
        gt_tube=(xmin, ymin, xmax, ymax)*sequence_length
        '''
        # encode = torch.zeros_like(prior_tube)
        p_x_min = prior_tube[0::4]
        p_y_min = prior_tube[1::4]
        p_x_max = prior_tube[2::4]
        p_y_max = prior_tube[3::4]
        prior_center_x = (p_x_min + p_x_max) / 2
        prior_center_y = (p_y_max + p_y_min) / 2
        prior_w = p_x_max - p_x_min
        prior_h = p_y_max - p_y_min

        g_x_min = gt_tube[0::4]
        g_y_min = gt_tube[1::4]
        g_x_max = gt_tube[2::4]
        g_y_max = gt_tube[3::4]
        gt_center_x = (g_x_min + g_x_max) / 2
        gt_center_y = (g_y_min + g_y_max) / 2
        gt_w = g_x_max - g_x_min
        gt_h = g_y_max - g_y_min

        encode[0::4] = (gt_center_x - prior_center_x) / prior_w / self.variance[0]
        encode[1::4] = (gt_center_y - prior_center_y) / prior_h / self.variance[1]
        encode[2::4] = torch.log(gt_w / prior_w) / self.variance[2]
        encode[3::4] = torch.log(gt_h / prior_h) / self.variance[3]

    def forward(self, output, ground_truth):
        loc_preds, conf_preds = output
        positive_samples_index_list, tubes_label, tubes_label_index, N = self.ACTMatchTube(self.tubes_init.all_tubes, ground_truth)
        loss_l = self.ACTGetLocLoss(loc_preds, positive_samples_index_list, self.tubes_init.all_tubes, ground_truth)
        tubes_loss = self.ACTComputeConfLoss(conf_preds, tubes_label)
        negtive_samples_index_list = self.ACTMineHardExamples(tubes_loss, positive_samples_index_list)
        loss_c = self.ACTGetConfLoss(conf_preds, positive_samples_index_list, negtive_samples_index_list, tubes_label_index)
        loss_l /= N
        # loss_c /= N
        return loss_l, loss_c


if __name__ == '__main__':
    import pickle

    with open("./debugfile.pkl", 'rb') as f:
        tube1 = pickle.load(f)
        tube2 = pickle.load(f)
    get_tube_overlap(tube1, tube2, 6)
