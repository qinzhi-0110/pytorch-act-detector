import pickle
import os
import torch
import torch.utils.data as data
import numpy as np
import cv2
from . import transforms
from torchvision.transforms import functional as F


class TubeDataset(data.Dataset):
    def __init__(self, DNAME, data_path, phase, modality, sequence_length):
        ground_truth_file = os.path.join(data_path, './cache/{}-GT.pkl'.format(DNAME))
        with open(ground_truth_file, 'rb') as fid:
            cache = pickle.load(fid, encoding='iso-8859-1')
        for k in cache:
            setattr(self, k, cache[k])
        self.MEAN = np.array([[[104, 117, 123]]], dtype=np.float32)
        self.DNAME = DNAME
        if DNAME == 'UCF101v2':
            self.image_format = '%05d.jpg'
        elif DNAME == 'UCFSports':
            self.image_format = '%06d.jpg'
        else:
            print("TubeDataset.DNAME value error!!")
            exit(-1)
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        self.expand = transforms.Expand(self.MEAN)
        self.modality = modality
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.image_shape = cache['resolution']
        if modality == 'rgb':
            self.rgb = True
        elif modality == 'flow':
            self.rgb = False
        else:
            print("dataset mode value error!")
            exit(-1)
        self.ground_tube_list = []
        self.videos_list = []
        self.label_list = []
        video_cnt = 0
        if phase == 'train':
            self.train = True
            for video in self.train_videos[0]:
                gttube = self.gttubes[video]
                if gttube.__len__() > 1:
                    print("tube_dataset.py: warning! gttube_list length > 1!!  maybe multi-labels in one video")
                    print("video:{}".format(video))
                    exit(-1)
                for key in gttube:
                    ts = gttube[key]
                    for t in ts:
                        for stf in range(t.shape[0]-self.sequence_length+1):
                            self.label_list += [key + 1]  # background label is 0
                            self.ground_tube_list += [t[stf:stf+self.sequence_length, :]]
                            self.videos_list += [video]
                        video_cnt += 1
                        # if video_cnt >= 8:
                        #     return
        elif phase == 'eval':
            self.train = False
            self.videos_list = self.test_videos[0]
            # self.videos_list = self.train_videos[0]
        else:
            print("dataset phase value error!")
            exit(-1)

    def __getitem__(self, index):
        # index = 10
        if self.rgb:
            root_path = os.path.join(self.data_path, 'Frames', self.videos_list[index])
        else:
            root_path = os.path.join(self.data_path, 'FlowBrox04', self.videos_list[index])
        image_list = []
        all_frames = os.listdir(root_path)
        nframes = all_frames.__len__()
        if self.train:
            gttube = self.ground_tube_list[index]
            ground_truth = np.zeros((1 + self.sequence_length * 4), dtype='float32')
            ground_truth[0] = self.label_list[index]
            for i in range(self.sequence_length):
                ground_truth[4*i+1:4*i+5] = gttube[i, 1:]
                image_path = os.path.join(root_path, self.image_format % min(int(gttube[i, 0]), nframes))
                im = cv2.imread(image_path)
                if im is None:
                    print("{}not found!!".format(image_path))
                    exit(-1)
                image_list += [im]

            image_list, ground_truth, _random_crop_data = transforms.random_crop(image_list, ground_truth)
            if (ground_truth[1::4] < ground_truth[3::4]).sum() != 6 or (ground_truth[2::4] < ground_truth[4::4]).sum() != 6:
                print("_random_crop_data saved!!")
                with open('./error_random_crop_data.pkl', 'wb') as f:
                    pickle.dump(index, f)
                    pickle.dump(image_list, f)
                    pickle.dump(ground_truth, f)
                    pickle.dump(_random_crop_data, f)
                exit(-1)

            image_list, ground_truth, _random_flip_data = transforms.random_flip(image_list, ground_truth)
            if (ground_truth[1::4] < ground_truth[3::4]).sum() != 6 or (ground_truth[2::4] < ground_truth[4::4]).sum() != 6:
                print("_random_crop_data saved!!")
                with open('./error_random_flip_data.pkl', 'wb') as f:
                    pickle.dump(index, f)
                    pickle.dump(image_list, f)
                    pickle.dump(ground_truth, f)
                    pickle.dump(_random_crop_data, f)
                    pickle.dump(_random_flip_data, f)
                exit(-1)

            image_list = self.color_jitter(image_list)
            image_list, ground_truth, _random_expand_data = self.expand(image_list, ground_truth)
            if (ground_truth[1::4] < ground_truth[3::4]).sum() != 6 or (
                    ground_truth[2::4] < ground_truth[4::4]).sum() != 6:
                print("_random_expand_data saved!!")
                with open('./error_random_expand_data.pkl', 'wb') as f:
                    pickle.dump(index, f)
                    pickle.dump(image_list, f)
                    pickle.dump(ground_truth, f)
                    pickle.dump(_random_crop_data, f)
                    pickle.dump(_random_flip_data, f)
                    pickle.dump(_random_expand_data, f)
                exit(-1)

            height_new, width_new, _ = image_list[0].shape
            ground_truth[1::4] = ground_truth[1::4] / width_new
            ground_truth[2::4] = ground_truth[2::4] / height_new
            ground_truth[3::4] = ground_truth[3::4] / width_new
            ground_truth[4::4] = ground_truth[4::4] / height_new
            for i in range(image_list.__len__()):
                image_list[i] = image_list[i] - self.MEAN
            image_data = np.concatenate(image_list, axis=2).astype('float32')
            image_data = cv2.resize(image_data, (300, 300), interpolation=cv2.INTER_LINEAR)
            image_data = np.transpose(image_data, (2, 0, 1))
            image_data = torch.from_numpy(image_data)
            ground_truth = torch.from_numpy(ground_truth)
        else:
            if self.rgb:
                for i in range(1, nframes+1):
                    image_path = os.path.join(root_path, self.image_format % i)
                    im = cv2.imread(image_path)
                    if im is None:
                        print("{}not found!!".format(image_path))
                        exit(-1)
                    im = cv2.resize(im, (300, 300), interpolation=cv2.INTER_LINEAR)
                    im = np.transpose(im - self.MEAN, (2, 0, 1))
                    image_list += [im]
                image_data = torch.from_numpy(np.vstack(image_list).astype('float32'))
                ground_truth = torch.Tensor([index])
            else:
                for i in range(1, nframes+1):
                    flow_path = os.path.join(root_path, self.image_format % i)
                    im = cv2.imread(flow_path)
                    im = cv2.resize(im, (300, 300), interpolation=cv2.INTER_LINEAR)
                    im = np.transpose(im - self.MEAN, (2, 0, 1))
                    # im = np.transpose(im, (2, 0, 1))
                    image_list += [im]
                image_data = torch.from_numpy(np.vstack(image_list).astype('float32'))
                ground_truth = torch.Tensor([index])
        return image_data, ground_truth

    def __len__(self):
        return self.videos_list.__len__()
        # return 48

    def get_test_videos(self):
        return self.test_videos

    def get_nframes(self):
        return self.nframes

    def get_resolution(self):
        return self.resolution

    def get_labels(self):
        return self.labels

    def get_gttubes(self):
        return self.gttubes


if __name__ == '__main__':
    import config
    args = config.Config()
    with open('./error_random_flip_data.pkl', 'wb') as f:
        index = pickle.load(f)
        image_list = pickle.load(f)
        ground_truth = pickle.load(f)
        _random_crop_data = pickle.load(f)
        _random_flip_data = pickle.load(f)
    train_dataset = TubeDataset(args.dataset, data_path=args.data_path, phase='train',
                                        modality=args.modality,
                                        sequence_length=6)
    a, b = train_dataset[index]
