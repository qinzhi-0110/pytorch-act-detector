import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np


class scale_norm(nn.Module):
    def __init__(self, channels):
        super(scale_norm, self).__init__()
        temp = torch.zeros(channels)
        temp.fill_(20)
        self.scale = nn.Parameter(temp.reshape(1, channels, 1, 1))

    def forward(self, input):
        output = F.normalize(input, p=2, dim=1)
        return output * self.scale


class SSD_NET(nn.Module):
    def __init__(self, dataset, frezze_init, num_classes=11, modality='rgb', k=6):
        super(SSD_NET, self).__init__()
        self.frezze_init = frezze_init
        self.k_frames = k
        self.dataset = dataset
        self.modality = modality
        if modality == 'rgb':
            self.rgb = True
        elif modality == 'flow':
            self.rgb = False
        else:
            print("modality value error!!")
            exit(-1)
        self.num_classes = num_classes
        if self.rgb:
            self.in_channels = 3
            self.layer_name = 'frame'
        else:
            self.in_channels = 15
            self.layer_name = 'flow'

        self.__setattr__('conv1_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu1_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv1_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu1_2_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('pool1_{}'.format(self.layer_name), nn.MaxPool2d(kernel_size=2, stride=2))
        #####################################

        self.__setattr__('conv2_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu2_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv2_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu2_2_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('pool2_{}'.format(self.layer_name), nn.MaxPool2d(kernel_size=2, stride=2))
        #####################################

        self.__setattr__('conv3_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu3_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv3_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu3_2_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv3_3_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu3_3_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('pool3_{}'.format(self.layer_name), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        #####################################

        self.__setattr__('conv4_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu4_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv4_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu4_2_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv4_3_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu4_3_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv4_3_norm_{}'.format(self.layer_name), scale_norm(512))
        self.__setattr__('pool4_{}'.format(self.layer_name), nn.MaxPool2d(kernel_size=2, stride=2))
        #####################################

        self.__setattr__('conv5_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu5_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv5_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu5_2_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv5_3_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.__setattr__('relu5_3_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('pool5_{}'.format(self.layer_name), nn.MaxPool2d(kernel_size=2, stride=1))
        #####################################

        self.__setattr__('fc_conv6_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1,
                                   dilation=6,
                                   padding=6))
        self.__setattr__('fc_relu6_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('fc_conv7_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, dilation=1,
                                   padding=0))
        self.__setattr__('fc_relu7_{}'.format(self.layer_name), nn.ReLU())
        #####################################

        self.__setattr__('conv6_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0))
        self.__setattr__('relu6_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv6_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))
        self.__setattr__('relu6_2_{}'.format(self.layer_name), nn.ReLU())

        self.__setattr__('conv7_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0))
        self.__setattr__('relu7_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv7_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))
        self.__setattr__('relu7_2_{}'.format(self.layer_name), nn.ReLU())

        self.__setattr__('conv8_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0))
        self.__setattr__('relu8_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv8_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0))
        self.__setattr__('relu8_2_{}'.format(self.layer_name), nn.ReLU())

        self.__setattr__('conv9_1_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0))
        self.__setattr__('relu9_1_{}'.format(self.layer_name), nn.ReLU())
        self.__setattr__('conv9_2_{}'.format(self.layer_name),
                         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0))
        self.__setattr__('relu9_2_{}'.format(self.layer_name), nn.ReLU())
        #####################################
        self.conv4_3_norm_loc_conv = nn.Conv2d(in_channels=3072, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_conf_conv = nn.Conv2d(in_channels=3072, out_channels=self.num_classes*4, kernel_size=3, stride=1, padding=1)

        self.fc_conv7_loc_conv = nn.Conv2d(in_channels=6144, out_channels=144, kernel_size=3, stride=1, padding=1)
        self.fc_conv7_conf_conv = nn.Conv2d(in_channels=6144, out_channels=self.num_classes*6, kernel_size=3, stride=1, padding=1)

        self.conv6_loc_conv = nn.Conv2d(in_channels=3072, out_channels=144, kernel_size=3, stride=1, padding=1)
        self.conv6_conf_conv = nn.Conv2d(in_channels=3072, out_channels=self.num_classes*6, kernel_size=3, stride=1, padding=1)

        self.conv7_loc_conv = nn.Conv2d(in_channels=1536, out_channels=144, kernel_size=3, stride=1, padding=1)
        self.conv7_conf_conv = nn.Conv2d(in_channels=1536, out_channels=self.num_classes*6, kernel_size=3, stride=1, padding=1)

        self.conv8_loc_conv = nn.Conv2d(in_channels=1536, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv8_conf_conv = nn.Conv2d(in_channels=1536, out_channels=self.num_classes*4, kernel_size=3, stride=1, padding=1)

        self.conv9_loc_conv = nn.Conv2d(in_channels=1536, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv9_conf_conv = nn.Conv2d(in_channels=1536, out_channels=self.num_classes*4, kernel_size=3, stride=1, padding=1)

    def copy_weights(self, conv_name_caffe, conv_name_pytorch, init_dict):
        my_conv = self.__getattr__(conv_name_pytorch)
        conv_name_caffe_list = init_dict[conv_name_caffe]
        caffe_weight = conv_name_caffe_list[0]
        my_conv.weight.data.copy_(torch.from_numpy(caffe_weight))
        if conv_name_caffe_list.__len__() == 2:
            caffe_bias = init_dict[conv_name_caffe][1]
        else:
            caffe_bias = np.zeros_like(my_conv.bias.data.numpy())
        my_conv.bias.data.copy_(torch.from_numpy(caffe_bias))

    def load_trained_weights(self, pkl_file):
        print('load trained weights^^^^^^')
        f = open(pkl_file, 'rb')
        init_dict = pickle.load(f, encoding='iso-8859-1')
        f.close()
        if self.rgb is True:
            cn = ''
        else:
            cn = 'flow'
        for j in range(1, 3):
            self.copy_weights('conv{}_1_stream{}{}'.format(j, 0, cn), 'conv{}_1_{}'.format(j, self.layer_name), init_dict)
            self.copy_weights('conv{}_2_stream{}{}'.format(j, 0, cn), 'conv{}_2_{}'.format(j, self.layer_name), init_dict)

        for j in range(3, 6):
            self.copy_weights('conv{}_1_stream{}{}'.format(j, 0, cn), 'conv{}_1_{}'.format(j, self.layer_name), init_dict)
            self.copy_weights('conv{}_2_stream{}{}'.format(j, 0, cn), 'conv{}_2_{}'.format(j, self.layer_name), init_dict)
            self.copy_weights('conv{}_3_stream{}{}'.format(j, 0, cn), 'conv{}_3_{}'.format(j, self.layer_name), init_dict)

        self.copy_weights('fc6_stream{}{}'.format(0, cn), 'fc_conv6_{}'.format(self.layer_name), init_dict)
        self.copy_weights('fc7_stream{}{}'.format(0, cn), 'fc_conv7_{}'.format(self.layer_name), init_dict)

        for j in range(6, 10):
            self.copy_weights('conv{}_1_stream{}{}'.format(j, 0, cn), 'conv{}_1_{}'.format(j, self.layer_name), init_dict)
            self.copy_weights('conv{}_2_stream{}{}'.format(j, 0, cn), 'conv{}_2_{}'.format(j, self.layer_name), init_dict)
        self.copy_weights('conv4_3_norm_concat_mbox_conf', 'conv4_3_norm_conf_conv', init_dict)
        self.copy_weights('conv4_3_norm_concat_mbox_loc', 'conv4_3_norm_loc_conv', init_dict)
        self.copy_weights('fc7_concat_mbox_loc', 'fc_conv7_loc_conv', init_dict)
        self.copy_weights('fc7_concat_mbox_conf', 'fc_conv7_conf_conv', init_dict)
        for j in range(6, 10):
            self.copy_weights('conv{}_2_concat_mbox_conf'.format(j), 'conv{}_conf_conv'.format(j), init_dict)
            self.copy_weights('conv{}_2_concat_mbox_loc'.format(j), 'conv{}_loc_conv'.format(j), init_dict)
        for j in range(0, 6):
            my_norm = self.__getattr__('conv4_3_norm_{}'.format(self.layer_name))
            caffe_weight = init_dict['conv4_3_norm_stream{}{}'.format(j, cn)][0]
            my_norm.scale.data.copy_(torch.from_numpy(caffe_weight).reshape(1, 512, 1, 1))
        torch.save(self.state_dict(), './pytorch-models/{}/{}-trained-model-{}-pytorch-single.pkl'.format(self.dataset, self.dataset, self.modality))
        print("pytorch model saved!!!")
        exit(0)

    def load_init_weights(self, pkl_file):
        print("load_init_weights")
        init_weights = torch.load(pkl_file)
        if self.rgb is True:
            cn = ''
        else:
            cn = 'flow'
        for j in range(1, 3):
            for k in range(1, 3):
                conv = self.__getattr__('conv{}_{}_{}'.format(j, k, self.layer_name))
                conv.weight.data.copy_(init_weights['conv{}_{}_stream{}{}.weight'.format(j, k, 0, cn)])
                conv.bias.data.copy_(init_weights['conv{}_{}_stream{}{}.bias'.format(j, k, 0, cn)])
        for j in range(3, 6):
            for k in range(1, 4):
                conv = self.__getattr__('conv{}_{}_{}'.format(j, k, self.layer_name))
                conv.weight.data.copy_(init_weights['conv{}_{}_stream{}{}.weight'.format(j, k, 0, cn)])
                conv.bias.data.copy_(init_weights['conv{}_{}_stream{}{}.bias'.format(j, k, 0, cn)])
        for j in range(6, 8):
            conv = self.__getattr__('fc_conv{}_{}'.format(j, self.layer_name))
            conv.weight.data.copy_(init_weights['fc{}_stream{}.weight'.format(j, 0)])
            conv.bias.data.copy_(init_weights['fc{}_stream{}.bias'.format(j, 0)])
        print("load ok!, save it!")
        torch.save(self.state_dict(), '/home/qzw/code/my-act-detector/pytorch-models/{}/{}-init-model-{}-pytorch-single.pkl'.format(self.dataset, self.dataset, self.modality))
        print("pytorch model saved!!!")
        exit(0)

    def forward(self, input):
        conv4_3_list = []
        fc_conv7_list = []
        conv6_list = []
        conv7_list = []
        conv8_list = []
        conv9_list = []

        for i in range(self.k_frames):
            output = input[:, self.in_channels * i:self.in_channels * (i + 1), :, :]
            output = self.__getattr__('conv1_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu1_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv1_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu1_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('pool1_{}'.format(self.layer_name))(output)

            output = self.__getattr__('conv2_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu2_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv2_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu2_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('pool2_{}'.format(self.layer_name))(output)

            output = self.__getattr__('conv3_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu3_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv3_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu3_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv3_3_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu3_3_{}'.format(self.layer_name))(output)
            output = self.__getattr__('pool3_{}'.format(self.layer_name))(output)

            output = self.__getattr__('conv4_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu4_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv4_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu4_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv4_3_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu4_3_{}'.format(self.layer_name))(output)
            conv4_3_list.append(self.__getattr__('conv4_3_norm_{}'.format(self.layer_name))(output))

            output = self.__getattr__('pool4_{}'.format(self.layer_name))(output)

            output = self.__getattr__('conv5_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu5_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv5_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu5_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv5_3_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu5_3_{}'.format(self.layer_name))(output)
            output = self.__getattr__('pool5_{}'.format(self.layer_name))(output)

            conv6 = self.__getattr__('fc_conv6_{}'.format(self.layer_name))
            conv6.dilation = (6 ** (i + 1), 6 ** (i + 1))
            conv6.padding = (6 ** (i + 1), 6 ** (i + 1))
            output = conv6(output)
            output = self.__getattr__('fc_relu6_{}'.format(self.layer_name))(output)
            output = self.__getattr__('fc_conv7_{}'.format(self.layer_name))(output)
            output = self.__getattr__('fc_relu7_{}'.format(self.layer_name))(output)
            fc_conv7_list.append(output)

            output = self.__getattr__('conv6_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu6_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv6_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu6_2_{}'.format(self.layer_name))(output)
            conv6_list.append(output)

            output = self.__getattr__('conv7_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu7_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv7_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu7_2_{}'.format(self.layer_name))(output)
            conv7_list.append(output)

            output = self.__getattr__('conv8_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu8_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv8_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu8_2_{}'.format(self.layer_name))(output)
            conv8_list.append(output)

            output = self.__getattr__('conv9_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu9_1_{}'.format(self.layer_name))(output)
            output = self.__getattr__('conv9_2_{}'.format(self.layer_name))(output)
            output = self.__getattr__('relu9_2_{}'.format(self.layer_name))(output)
            conv9_list.append(output)
        conv4_3_fm = torch.cat(conv4_3_list, dim=1).contiguous()
        fc_conv7_fm = torch.cat(fc_conv7_list, dim=1).contiguous()
        conv6_fm = torch.cat(conv6_list, dim=1).contiguous()
        conv7_fm = torch.cat(conv7_list, dim=1).contiguous()
        conv8_fm = torch.cat(conv8_list, dim=1).contiguous()
        conv9_fm = torch.cat(conv9_list, dim=1).contiguous()

        conv4_3_norm_localization = self.conv4_3_norm_loc_conv(conv4_3_fm)
        conv4_3_norm_localization = conv4_3_norm_localization.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.k_frames*4)
        conv4_3_norm_confidence = self.conv4_3_norm_conf_conv(conv4_3_fm)
        conv4_3_norm_confidence = conv4_3_norm_confidence.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.num_classes)
        # conv4_3_norm_confidence = F.softmax(conv4_3_norm_confidence, dim=-1)

        fc_conv7_localization = self.fc_conv7_loc_conv(fc_conv7_fm)
        fc_conv7_localization = fc_conv7_localization.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.k_frames*4)
        fc_conv7_confidence = self.fc_conv7_conf_conv(fc_conv7_fm)
        fc_conv7_confidence = fc_conv7_confidence.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.num_classes)
        # fc_conv7_confidence = F.softmax(fc_conv7_confidence, dim=-1)

        conv6_localization = self.conv6_loc_conv(conv6_fm)
        conv6_localization = conv6_localization.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.k_frames*4)
        conv6_confidence = self.conv6_conf_conv(conv6_fm)
        conv6_confidence = conv6_confidence.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.num_classes)
        # conv6_confidence = F.softmax(conv6_confidence, dim=-1)

        conv7_localization = self.conv7_loc_conv(conv7_fm)
        conv7_localization = conv7_localization.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.k_frames*4)
        conv7_confidence = self.conv7_conf_conv(conv7_fm)
        conv7_confidence = conv7_confidence.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.num_classes)
        # conv7_confidence = F.softmax(conv7_confidence, dim=-1)

        conv8_localization = self.conv8_loc_conv(conv8_fm)
        conv8_localization = conv8_localization.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.k_frames*4)
        conv8_confidence = self.conv8_conf_conv(conv8_fm)
        conv8_confidence = conv8_confidence.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.num_classes)
        # conv8_confidence = F.softmax(conv8_confidence, dim=-1)

        conv9_localization = self.conv9_loc_conv(conv9_fm)
        conv9_localization = conv9_localization.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.k_frames*4)
        conv9_confidence = self.conv9_conf_conv(conv9_fm)
        conv9_confidence = conv9_confidence.permute(0, 2, 3, 1).contiguous().view(input.shape[0], -1, self.num_classes)
        # conv9_confidence = F.softmax(conv9_confidence, dim=-1)

        loc_preds = torch.cat([conv4_3_norm_localization, fc_conv7_localization, conv6_localization, conv7_localization,
                          conv8_localization, conv9_localization], dim=1)
        conf_preds = torch.cat([conv4_3_norm_confidence, fc_conv7_confidence, conv6_confidence, conv7_confidence,
                           conv8_confidence, conv9_confidence], dim=1)
        return loc_preds, conf_preds

    def get_feature_map(self, input, conv6_dilation):
        conv6 = self.__getattr__('fc_conv6_{}'.format(self.layer_name))
        conv6.dilation = conv6_dilation
        conv6.padding = conv6_dilation
        output = self.__getattr__('conv1_1_{}'.format(self.layer_name))(input)
        output = self.__getattr__('relu1_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv1_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu1_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('pool1_{}'.format(self.layer_name))(output)

        output = self.__getattr__('conv2_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu2_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv2_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu2_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('pool2_{}'.format(self.layer_name))(output)

        output = self.__getattr__('conv3_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu3_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv3_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu3_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv3_3_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu3_3_{}'.format(self.layer_name))(output)
        output = self.__getattr__('pool3_{}'.format(self.layer_name))(output)

        output = self.__getattr__('conv4_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu4_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv4_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu4_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv4_3_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu4_3_{}'.format(self.layer_name))(output)
        conv4_3 = self.__getattr__('conv4_3_norm_{}'.format(self.layer_name))(output)

        output = self.__getattr__('pool4_{}'.format(self.layer_name))(output)

        output = self.__getattr__('conv5_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu5_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv5_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu5_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv5_3_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu5_3_{}'.format(self.layer_name))(output)
        output = self.__getattr__('pool5_{}'.format(self.layer_name))(output)

        output = self.__getattr__('fc_conv6_{}'.format(self.layer_name))(output)
        output = self.__getattr__('fc_relu6_{}'.format(self.layer_name))(output)
        output = self.__getattr__('fc_conv7_{}'.format(self.layer_name))(output)
        output = self.__getattr__('fc_relu7_{}'.format(self.layer_name))(output)
        fc_conv7 = output

        output = self.__getattr__('conv6_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu6_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv6_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu6_2_{}'.format(self.layer_name))(output)
        conv6 = output

        output = self.__getattr__('conv7_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu7_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv7_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu7_2_{}'.format(self.layer_name))(output)
        conv7 = output

        output = self.__getattr__('conv8_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu8_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv8_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu8_2_{}'.format(self.layer_name))(output)
        conv8 = output

        output = self.__getattr__('conv9_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu9_1_{}'.format(self.layer_name))(output)
        output = self.__getattr__('conv9_2_{}'.format(self.layer_name))(output)
        output = self.__getattr__('relu9_2_{}'.format(self.layer_name))(output)
        conv9 = output
        return conv4_3, fc_conv7, conv6, conv7, conv8, conv9

    def get_loc_conf(self, conv4_3_data, fc_conv7_data, conv6_data, conv7_data, conv8_data, conv9_data):
        conv4_3_norm_localization = self.conv4_3_norm_loc_conv(conv4_3_data.cuda())
        conv4_3_norm_localization = conv4_3_norm_localization.permute(0, 2, 3, 1).contiguous().view(
            conv4_3_data.shape[0],
            -1,
            self.k_frames * 4)
        conv4_3_norm_confidence = self.conv4_3_norm_conf_conv(conv4_3_data.cuda())
        conv4_3_norm_confidence = conv4_3_norm_confidence.permute(0, 2, 3, 1).contiguous().view(conv4_3_data.shape[0],
                                                                                                -1,
                                                                                                self.num_classes)
        fc_conv7_localization = self.fc_conv7_loc_conv(fc_conv7_data.cuda())
        fc_conv7_localization = fc_conv7_localization.permute(0, 2, 3, 1).contiguous().view(fc_conv7_data.shape[0], -1,
                                                                                            self.k_frames * 4)

        fc_conv7_confidence = self.fc_conv7_conf_conv(fc_conv7_data.cuda())
        fc_conv7_confidence = fc_conv7_confidence.permute(0, 2, 3, 1).contiguous().view(fc_conv7_data.shape[0], -1,
                                                                                        self.num_classes)

        conv6_localization = self.conv6_loc_conv(conv6_data.cuda())
        conv6_localization = conv6_localization.permute(0, 2, 3, 1).contiguous().view(conv6_data.shape[0], -1,
                                                                                      self.k_frames * 4)
        conv6_confidence = self.conv6_conf_conv(conv6_data.cuda())
        conv6_confidence = conv6_confidence.permute(0, 2, 3, 1).contiguous().view(conv6_data.shape[0], -1,
                                                                                  self.num_classes)

        conv7_localization = self.conv7_loc_conv(conv7_data.cuda())
        conv7_localization = conv7_localization.permute(0, 2, 3, 1).contiguous().view(conv7_data.shape[0], -1,
                                                                                      self.k_frames * 4)
        conv7_confidence = self.conv7_conf_conv(conv7_data.cuda())
        conv7_confidence = conv7_confidence.permute(0, 2, 3, 1).contiguous().view(conv7_data.shape[0], -1,
                                                                                  self.num_classes)

        conv8_localization = self.conv8_loc_conv(conv8_data.cuda())
        conv8_localization = conv8_localization.permute(0, 2, 3, 1).contiguous().view(conv8_data.shape[0], -1,
                                                                                      self.k_frames * 4)
        conv8_confidence = self.conv8_conf_conv(conv8_data.cuda())
        conv8_confidence = conv8_confidence.permute(0, 2, 3, 1).contiguous().view(conv8_data.shape[0], -1,
                                                                                  self.num_classes)

        conv9_localization = self.conv9_loc_conv(conv9_data.cuda())
        conv9_localization = conv9_localization.permute(0, 2, 3, 1).contiguous().view(conv9_data.shape[0], -1,
                                                                                      self.k_frames * 4)
        conv9_confidence = self.conv9_conf_conv(conv9_data.cuda())
        conv9_confidence = conv9_confidence.permute(0, 2, 3, 1).contiguous().view(conv9_data.shape[0], -1,
                                                                                  self.num_classes)
        loc_preds = torch.cat(
            [conv4_3_norm_localization, fc_conv7_localization, conv6_localization, conv7_localization,
             conv8_localization, conv9_localization], dim=1)
        conf_preds = torch.cat([conv4_3_norm_confidence, fc_conv7_confidence, conv6_confidence, conv7_confidence,
                                conv8_confidence, conv9_confidence], dim=1)
        return loc_preds, conf_preds

    def train(self, mode=True):
        super(SSD_NET, self).train(mode)
        for m in self.modules():
            ps = list(m.parameters())
            for p in ps:
                p.requires_grad = True
        self.conv4_3_norm_conf_conv.bias.data.fill_(0)
        self.conv4_3_norm_conf_conv.bias.requires_grad = False
        self.conv4_3_norm_loc_conv.bias.data.fill_(0)
        self.conv4_3_norm_loc_conv.bias.requires_grad = False
        if self.frezze_init:
            self.frezze_init_func(freeze_norm_layer=False)

    def eval(self):
        super(SSD_NET, self).eval()
        for m in self.modules():
            ps = list(m.parameters())
            for p in ps:
                p.requires_grad = False

    def get_optim_policies(self):
        parameters_list = []
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                ps = list(m.parameters())
                if ps[0].requires_grad:
                    parameters_list.append(ps[0])
                if ps[1].requires_grad:
                    parameters_list.append(ps[1])
            elif isinstance(m, scale_norm):
                ps = list(m.parameters())
                if ps[0].requires_grad:
                    parameters_list.append(ps[0])
        return parameters_list

    def get_loc_conf_optim_policies(self):
        parameters_list = []
        parameters_list.append(self.conv4_3_norm_loc_conv.weight)
        parameters_list.append(self.conv4_3_norm_conf_conv.weight)

        parameters_list.append(self.fc_conv7_loc_conv.weight)
        parameters_list.append(self.fc_conv7_loc_conv.bias)
        parameters_list.append(self.fc_conv7_conf_conv.weight)
        parameters_list.append(self.fc_conv7_conf_conv.bias)

        parameters_list.append(self.conv6_loc_conv.weight)
        parameters_list.append(self.conv6_loc_conv.bias)
        parameters_list.append(self.conv6_conf_conv.weight)
        parameters_list.append(self.conv6_conf_conv.bias)

        parameters_list.append(self.conv7_loc_conv.weight)
        parameters_list.append(self.conv7_loc_conv.bias)
        parameters_list.append(self.conv7_conf_conv.weight)
        parameters_list.append(self.conv7_conf_conv.bias)

        parameters_list.append(self.conv8_loc_conv.weight)
        parameters_list.append(self.conv8_loc_conv.bias)
        parameters_list.append(self.conv8_conf_conv.weight)
        parameters_list.append(self.conv8_conf_conv.bias)

        parameters_list.append(self.conv9_loc_conv.weight)
        parameters_list.append(self.conv9_loc_conv.bias)
        parameters_list.append(self.conv9_conf_conv.weight)
        parameters_list.append(self.conv9_conf_conv.bias)
        return parameters_list

    def get_vgg_optim_policies(self):
        parameters_list = []
        conv = self.__getattr__('conv1_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv1_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv2_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv2_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv3_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv3_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv3_3_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv4_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv4_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv4_3_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv4_3_norm_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv5_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv5_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv5_3_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        return parameters_list

    def get_ssd_optim_policies(self):
        parameters_list = []
        conv = self.__getattr__('fc_conv6_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('fc_conv7_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv6_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv6_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv7_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv7_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv8_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv8_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv9_1_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        conv = self.__getattr__('conv9_2_{}'.format(self.layer_name))
        parameters_list.append(conv.weight)
        parameters_list.append(conv.bias)
        return parameters_list

    def frezze_init_func(self, freeze_norm_layer=False):
        m = self.__getattr__('conv1_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv1_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv2_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv2_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv3_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv3_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv3_3_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv4_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv4_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv4_3_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

        m = self.__getattr__('conv5_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv5_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv5_3_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('fc_conv6_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('fc_conv7_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        if freeze_norm_layer:
            m = self.__getattr__('conv4_3_norm_{}'.format(self.layer_name))
            m.eval()
            m.scale.requires_grad = False

    def freeze_ssd(self, freeze_norm_layer):
        self.frezze_vgg(freeze_norm_layer)
        m = self.__getattr__('fc_conv6_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('fc_conv7_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv6_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv6_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

        m = self.__getattr__('conv7_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv7_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

        m = self.__getattr__('conv8_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv8_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

        m = self.__getattr__('conv9_1_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m = self.__getattr__('conv9_2_{}'.format(self.layer_name))
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

if __name__ == '__main__':
    # with open('/media/main/_sdc/qzw/ACT-Detector/my-act-detector/caffe-models/initialization_VGG_ILSVRC16_K6_RGB.pkl', 'rb') as f:
    #     initialization_dict = pickle.load(f)
    #     f.close()
    # rgb_net = SSD_NET(num_classes=25, rgb=True)

    # rgb_net.load_init_weights(
    #     './caffe-models/UCF101v2/RGB-UCF101v2-numpy.pkl')
    # torch.save(rgb_net.state_dict(), 'RGB-UCF101v2-pytorch.pkl')
    # print("RGB OK!!!")

    flow_net = SSD_NET(num_classes=25, rgb=False)
    flow_net.load_init_weights(
        './caffe-models/UCF101v2/FLOW5-UCF101v2-numpy.pkl')
    torch.save(flow_net.state_dict(), 'FLOW5-UCF101v2-pytorch.pkl')
    print("FLOW5 OK!!!")



