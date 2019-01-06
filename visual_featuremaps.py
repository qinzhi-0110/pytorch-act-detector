import torch
from layers import ssd
import numpy as np
import cv2
import os

def main():
    dataset = 'UCFSports'
    modality = 'rgb'
    data_path = "/mnt/data/qzw/data/UCFSports/"
    feature_map_path = "/mnt/data/qzw/result/pytorch-act-detector/{}/feature_maps/".format(dataset)
    MEAN = np.array([[[104, 117, 123]]], dtype=np.float32)
    test_net = ssd.SSD_NET(dataset=dataset, frezze_init=True, num_classes=11,
                            modality=modality)

    data_dict = torch.load("/mnt/data/qzw/model/pytorch-act-detector/{}/best-{}-cpu-0.8601.pkl" .format(dataset, modality))
    # net_state_dict = {}
    # for key in data_dict['net_state_dict']:
    #     if 'module.' in key:
    #         new_key = key.replace('module.', '')
    #     else:
    #         new_key = key
    #     net_state_dict[new_key] = data_dict['net_state_dict'][key]
    test_net.load_state_dict(data_dict)

    image = cv2.imread(os.path.join(data_path, "Frames", '084', '%06d.jpg' % 1))
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)
    image = np.transpose(image - MEAN, (2, 0, 1))[None, :, :, :]
    image = torch.from_numpy(image.astype('float32'))

    conv6_dilation = (6, 6)
    conv6 = test_net.__getattr__('fc_conv6_{}'.format(test_net.layer_name))
    conv6.dilation = conv6_dilation
    conv6.padding = conv6_dilation

    output = image
    for name, layer in test_net._modules.items():
        output = layer(output)
        if 'conv' in name or '9' in name:
            continue
        save_path = os.path.join(feature_map_path, name)
        if os.path.exists(save_path) is not True:
            os.mkdir(save_path)
        feature_maps = output.squeeze().detach().numpy()
        for i in range(feature_maps.shape[0]):
            feature_map = feature_maps[i, :, :][:, :, None]
            if np.max(feature_map) > 0.001:
                feature_map = feature_map*255.0/np.max(feature_map)
            feature_map = cv2.resize(feature_map, (300, 300), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_path, "%03d.jpg" % (i+1)), feature_map)





if __name__ == '__main__':
    main()
