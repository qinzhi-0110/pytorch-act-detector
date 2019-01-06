import numpy as np
import torch
from utils import prior_tubes
from utils import map_eval
from data import dataset
import multiprocessing
import time
import torch.nn.functional as F
import pickle


def data_handle_and_save_process(all_frame_boxes_dict, video_index, conf_preds, decode_video_tubes, num_class,
                                 sequence_length, height, width):
    all_frame_boxes_list = []
    start_frame = 0
    frame_boxes = {}
    for batch in range(conf_preds.shape[0]):
        start_frame += 1
        nms_tubes_blist, nms_scores, nms_label_list = prior_tubes.apply_nms(conf_preds[batch, :],
                                                                            decode_video_tubes[batch, :],
                                                                            nms_threshold=0.45,
                                                                            num_class=num_class)
        if nms_scores.__len__() > 0:
            tt1 = (torch.Tensor(nms_label_list) - 1).view(-1, 1).numpy()
            tt2 = torch.Tensor(nms_scores).view(-1, 1).numpy()
            tt3 = np.vstack([tt.view(1, -1).cpu().numpy() for tt in nms_tubes_blist])
            best_tube = np.hstack([tt1, tt2, tt3])
        else:
            best_tube = np.array([])
        for m in range(best_tube.shape[0]):
            for n in range(sequence_length):
                if (n + start_frame) not in frame_boxes:
                    frame_boxes[n + start_frame] = []
                frame_boxes[n + start_frame] += [
                    best_tube[m, np.array([0, 1, 2 + 4 * n, 3 + 4 * n, 4 + 4 * n, 5 + 4 * n])]]
    # print("video:{}/{}ok!".format(video_index, eval_dataset.__len__()), "\ttime:", time.time() - time_start,
    #       "frame:{}".format(nframes))
    for frame_index in frame_boxes:
        frame_label = {}  # 记录了当前帧上各个label的所有框
        for bb in frame_boxes[frame_index]:
            if bb[0] not in frame_label:
                frame_label[bb[0]] = []
            frame_label[bb[0]] += [bb[1:]]
        for tt in frame_label:
            idx = map_eval.nms_class(frame_label[tt], nms_threshold=0.3)
            for id in idx:
                all_frame_boxes_list += [np.hstack([np.array([video_index, frame_index, tt]),
                                                    frame_label[tt][id] * np.array([1, width, height, width, height])])]
    all_frame_boxes_dict[video_index] = all_frame_boxes_list
    print("video_index:{} OK!!".format(video_index))


def eval_rgb_or_flow(model, eval_dataset, eval_dataloader, args, GEN_NUM):
    if args.dataset == 'UCF101v2':
        num_class = 25
    elif args.dataset == 'UCFSports':
        num_class = 11
    else:
        num_class = 0
        print("No dataset name {}".format(args.dataset))
        exit(0)
    rgb = args.modality == 'rgb'
    use_gpu = args.use_gpu
    variance = args.variance
    # model = ssd_net_ucf101.SSD_NET(dataset=args.dataset, num_classes=num_class, modality=args.modality)
    # if args.reinit_all:
    #     print("reinit all data!!!")
    #     # model.load_trained_weights('/home/qzw/code/my-act-detector/caffe-models/UCFSports/FLOW5-UCFSports.pkl')
    #     # pytorch_model = '/home/qzw/code/my-act-detector/pytorch-models/{}/{}-trained-model-{}-pytorch-single.pkl'.format(args.dataset, args.dataset, args.modality)
    #     # model.load_state_dict(torch.load(pytorch_model))
    #     # GEN_NUM = 0
    #     pytorch_model = '/home/qzw/code/my-act-detector-12-13/my_trained_pytorch_model_{}-{}.pkl'.format(args.dataset, args.modality)
    #     data_dict = torch.load(pytorch_model)
    #     GEN_NUM = data_dict['gen_num']
    #     net_state_dict = {}
    #     for key in data_dict['net_state_dict']:
    #         if 'module.' in key:
    #             new_key = key.replace('module.', '')
    #         else:
    #             new_key = key
    #         net_state_dict[new_key] = data_dict['net_state_dict'][key]
    #     model.load_state_dict(net_state_dict)
    # if use_gpu:
    #     model.cuda()
    #     # model = torch.nn.DataParallel(model).cuda()
    model.eval()
    eval_dataset = dataset.TubeDataset(args.dataset, data_path=args.data_path, phase='eval',
                                            modality=args.modality, sequence_length=6)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                                  num_workers=8, pin_memory=True)
    tubes_init = prior_tubes.RGB_TUBES(phase='eval', use_gpu=use_gpu, variance=variance, sequence_length=6)
    manager = multiprocessing.Manager()
    all_frame_boxes_dict = manager.dict()
    pool = multiprocessing.Pool(processes=16)
    resolution = eval_dataset.get_resolution()
    start_time = time.time()
    nframes_sum = 0
    for i, (input, target) in enumerate(eval_dataloader):
        video_index = i + 1
        nframes = int(input.shape[1] / 3)
        print("GEN_NUM:{}  video_index:{}/{} start!! frame num:{}  nframes_sum:{} fps:{}".format(GEN_NUM, video_index,
                                                                                                 eval_dataset.__len__(),
                                                                                                 nframes, nframes_sum,
                                                                                                 nframes_sum / (
                                                                                                             time.time() - start_time)))
        nframes_sum += nframes
        height, width = resolution[eval_dataset.videos_list[int(target[0, 0])]]
        if use_gpu:
            input = input.cuda()
        d36_dict = {}
        d36_dict['conv4_3'] = [0]
        d36_dict['fc_conv7'] = [0]
        d36_dict['conv6'] = [0]
        d36_dict['conv7'] = [0]
        d36_dict['conv8'] = [0]
        d36_dict['conv9'] = [0]
        conf_preds_list = []
        decode_video_tubes_list = []
        for d in range(1, args.sequence_length - 1):
            conv4_3_d36, fc_conv7_d36, conv6_d36, conv7_d36, conv8_d36, conv9_d36 = model.get_feature_map(
                input[0, 3 * d:3 * (1 + d), :, :].unsqueeze(0), (36, 36))
            d36_dict['conv4_3'] += [conv4_3_d36]
            d36_dict['fc_conv7'] += [fc_conv7_d36]
            d36_dict['conv6'] += [conv6_d36]
            d36_dict['conv7'] += [conv7_d36]
            d36_dict['conv8'] += [conv8_d36]
            d36_dict['conv9'] += [conv9_d36]

        for frame_index in range(nframes - args.sequence_length + 1):
            if rgb:
                conv4_3_d6, fc_conv7_d6, conv6_d6, conv7_d6, conv8_d6, conv9_d6 = model.get_feature_map(
                    input[0, 3 * frame_index:3 * (frame_index + 1), :, :].unsqueeze(0), (6, 6))
                conv4_3_d36, fc_conv7_d36, conv6_d36, conv7_d36, conv8_d36, conv9_d36 = model.get_feature_map(
                    input[0, 3 * (frame_index + args.sequence_length - 1):3 * (frame_index + args.sequence_length), :,
                    :].unsqueeze(0), (36, 36))
            else:
                conv4_3_d6, fc_conv7_d6, conv6_d6, conv7_d6, conv8_d6, conv9_d6 = model.get_feature_map(
                    input[0, 3 * frame_index:3 * (frame_index + args.sequence_length - 1), :, :].unsqueeze(0), (6, 6))
                conv4_3_d36, fc_conv7_d36, conv6_d36, conv7_d36, conv8_d36, conv9_d36 = model.get_feature_map(
                    torch.cat([input[0, 3 * min(frame_index + args.sequence_length - 1 + ff, nframes - 1):3 * (
                            min(frame_index + args.sequence_length - 1 + ff, nframes - 1) + 1), :, :].unsqueeze(0) for
                               ff in range(args.sequence_length - 1)], dim=1), (36, 36))
            d36_dict['conv4_3'] += [conv4_3_d36]
            d36_dict['fc_conv7'] += [fc_conv7_d36]
            d36_dict['conv6'] += [conv6_d36]
            d36_dict['conv7'] += [conv7_d36]
            d36_dict['conv8'] += [conv8_d36]
            d36_dict['conv9'] += [conv9_d36]
            d36_dict['conv4_3'][frame_index] = 0
            d36_dict['fc_conv7'][frame_index] = 0
            d36_dict['conv6'][frame_index] = 0
            d36_dict['conv7'][frame_index] = 0
            d36_dict['conv8'][frame_index] = 0
            d36_dict['conv9'][frame_index] = 0
            conv4_3_data = torch.cat([conv4_3_d6] + [d36_dict['conv4_3'][ff] for ff in
                                                     range(frame_index + 1, frame_index + args.sequence_length)],
                                     dim=1)
            fc_conv7_data = torch.cat([fc_conv7_d6] + [d36_dict['fc_conv7'][ff] for ff in
                                                       range(frame_index + 1, frame_index + args.sequence_length)],
                                      dim=1)
            conv6_data = torch.cat([conv6_d6] + [d36_dict['conv6'][ff] for ff in
                                                 range(frame_index + 1, frame_index + args.sequence_length)],
                                   dim=1)
            conv7_data = torch.cat([conv7_d6] + [d36_dict['conv7'][ff] for ff in
                                                 range(frame_index + 1, frame_index + args.sequence_length)],
                                   dim=1)
            conv8_data = torch.cat([conv8_d6] + [d36_dict['conv8'][ff] for ff in
                                                 range(frame_index + 1, frame_index + args.sequence_length)],
                                   dim=1)
            conv9_data = torch.cat([conv9_d6] + [d36_dict['conv9'][ff] for ff in
                                                 range(frame_index + 1, frame_index + args.sequence_length)],
                                   dim=1)
            loc_preds, conf_preds = model.get_loc_conf(conv4_3_data, fc_conv7_data, conv6_data, conv7_data,
                                                       conv8_data, conv9_data)
            conf_preds = F.softmax(conf_preds, dim=-1)
            decode_video_tubes = prior_tubes.decode_tubes(tubes_init, loc_preds)
            conf_preds_list += [conf_preds.cpu()]
            decode_video_tubes_list += [decode_video_tubes.cpu()]
        conf_preds = torch.cat(conf_preds_list, dim=0)
        decode_video_tubes = torch.cat(decode_video_tubes_list, dim=0)
        # data_handle_and_save_process(all_frame_boxes_dict, video_index, conf_preds, decode_video_tubes, num_class,
        #                              args.sequence_length, height, width)
        pool.apply_async(data_handle_and_save_process, (all_frame_boxes_dict, video_index, conf_preds,
                                                        decode_video_tubes, num_class, args.sequence_length,
                                                        height, width, ))
    print("waiting calc!!")
    pool.close()
    pool.join()
    print("all ok!!")
    all_frame_boxes_list = []
    for key in all_frame_boxes_dict:
        all_frame_boxes_list += all_frame_boxes_dict[key]
    with open(args.all_frame_boxes_list_result, "wb") as file:
        pickle.dump(all_frame_boxes_list, file)
    return map_eval.calc_pr(all_frame_boxes_list, eval_dataset)


if __name__ == '__main__':
    import config
    from layers import ssd
    args = config.Config()
    if args.dataset == 'UCF101v2':
        num_class = 25
    elif args.dataset == 'UCFSports':
        num_class = 11
    else:
        num_class = 0
        print("No dataset name {}".format(args.dataset))
        exit(0)
    eval_net = ssd.SSD_NET(dataset=args.dataset, frezze_init=args.freeze_init, num_classes=num_class,
                            modality=args.modality)
    data_dict = torch.load("/mnt/data/qzw/model/pytorch-act-detector/{}/best-rgb-0.8601.pkl".format(args.dataset))
    net_state_dict = {}
    for key in data_dict['net_state_dict']:
        if 'module.' in key:
            new_key = key.replace('module.', '')
        else:
            new_key = key
        net_state_dict[new_key] = data_dict['net_state_dict'][key]
    eval_net.load_state_dict(net_state_dict)
    if args.use_gpu:
        eval_net = eval_net.cuda()
    mmap = eval_rgb_or_flow(model=eval_net, eval_dataset=None, eval_dataloader=None, args=args,
                                 GEN_NUM=data_dict['gen_num'])
