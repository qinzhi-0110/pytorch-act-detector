import numpy as np
import random
import cv2
from torchvision.transforms import functional as F
from PIL import Image


def random_flip(image_list, target):
    if np.random.random() < 0.5:
        height, width, _ = image_list[0].shape
        for i in range(image_list.__len__()):
            image_list[i] = np.flip(image_list[i], axis=1)
        xmin_new = width - target[3::4]
        target[3::4] = width - target[1::4]
        target[1::4] = xmin_new
    return image_list, target


def random_crop(image_list, target):
    target = target.copy()
    scale = 0.5
    height, width, _ = image_list[0].shape
    gt_w = target[3::4] - target[1::4]
    gt_h = target[4::4] - target[2::4]
    gt_area = gt_w * gt_h
    gt_area = gt_area.sum()
    while True:
        xmin_crop = (1 - scale) * np.random.random()
        # ymin_crop = (1 - scale) * np.random.random()
        ymin_crop = xmin_crop
        xmax_crop = xmin_crop + (1 - xmin_crop - scale) * np.random.random() + scale
        # ymax_crop = ymin_crop + (1 - ymin_crop - scale) * np.random.random() + scale
        ymax_crop = xmax_crop
        xmin_crop, ymin_crop, xmax_crop, ymax_crop = int(width*xmin_crop), int(height*ymin_crop), int(width*xmax_crop), int(height*ymax_crop)

        xmin_cross = np.maximum(target[1::4], xmin_crop)
        ymin_cross = np.maximum(target[2::4], ymin_crop)
        xmax_cross = np.minimum(target[3::4], xmax_crop)
        ymax_cross = np.minimum(target[4::4], ymax_crop)

        cross_w = xmax_cross - xmin_cross
        cross_h = ymax_cross - ymin_cross
        if (cross_w < 0).sum() > 0:
            continue
        if (cross_h < 0).sum() > 0:
            continue
        cross_area = cross_w * cross_h
        cross_area = cross_area.sum()
        if cross_area / gt_area < 0.8:
            continue
        target[1::4] = xmin_cross - xmin_crop
        target[2::4] = ymin_cross - ymin_crop
        target[3::4] = xmax_cross - xmin_crop
        target[4::4] = ymax_cross - ymin_crop
        break

    image_list_new = [image_list[i][ymin_crop:ymax_crop+1, xmin_crop:xmax_crop+1, :] for i in range(len(image_list))]

    return image_list_new, target


class ColorJitter(object):
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(lambda img: F.adjust_brightness(img, brightness_factor))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(lambda img: F.adjust_contrast(img, contrast_factor))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(lambda img: F.adjust_saturation(img, saturation_factor))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(lambda img: F.adjust_hue(img, hue_factor))
        np.random.shuffle(transforms)
        return transforms

    def __call__(self, img_list):
        transforms = self.get_params(self.brightness, self.contrast,
                                     self.saturation, self.hue)
        for i in range(img_list.__len__()):
            img = img_list[i][..., -1::-1]  # bgr2rgb
            img = Image.fromarray(np.uint8(img))
            for t in transforms:
                img = t(img)
            img = np.asarray(img)
            img_list[i] = img[..., -1::-1]  # rgb2bgr
        return img_list


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = F.normalize(tensor, self.mean, self.std)
        return tensor


class Expand(object):
    def __init__(self, mean):
        self.expand_prob = 0.5
        self.max_expand_ratio = 4.0
        self.mean = mean

    def __call__(self, image_list, ground_truth):
        out_image_list = image_list
        if np.random.random() < self.expand_prob:
            expand_ratio = np.random.uniform(1, self.max_expand_ratio)
            ori_h, ori_w, _ = image_list[0].shape
            new_h, new_w = int(ori_h*expand_ratio), int(ori_w*expand_ratio)
            out_image_list = [(np.zeros((new_h, new_w, 3), dtype=np.float32) + self.mean) for i in range(len(image_list))]
            h_off, w_off = int(np.floor(new_h - ori_h)), int(np.floor(new_w - ori_w))
            for i in range(len(image_list)):
                out_image_list[i][h_off:h_off+ori_h, w_off:w_off+ori_w] = image_list[i]
            ground_truth[1:] += np.array([w_off, h_off, w_off, h_off]*len(image_list), dtype=np.float32)
        return out_image_list, ground_truth


def PCA_Jittering(img):
    img_size = img.size / 3
    # print(img.size, img_size)
    img1 = img.reshape(int(img_size), 3)
    img1 = np.transpose(img1)
    img_cov = np.cov([img1[0], img1[1], img1[2]])
    # 计算矩阵特征向量
    lamda, p = np.linalg.eig(img_cov)

    p = np.transpose(p)
    # 生成正态分布的随机数
    alpha1 = random.normalvariate(0, 0.05)
    alpha2 = random.normalvariate(0, 0.05)
    alpha3 = random.normalvariate(0, 0.05)
    v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))  # 加入扰动
    add_num = np.dot(p, v)
    img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])
    img2 = np.swapaxes(img2, 0, 2)
    img2 = np.swapaxes(img2, 0, 1)
    img2[img2 < 0] = 0
    img2[img2 > 255] = 255
    # max_t = np.max(img2[:, :, 0])
    # min_t = np.min(img2[:, :, 0])
    # img2[:, :, 0] = 255 / (max_t - min_t) * (img2[:, :, 0] - min_t)
    #
    # max_t = np.max(img2[:, :, 1])
    # min_t = np.min(img2[:, :, 1])
    # img2[:, :, 1] = 255 / (max_t - min_t) * (img2[:, :, 1] - min_t)
    #
    # max_t = np.max(img2[:, :, 2])
    # min_t = np.min(img2[:, :, 2])
    # img2[:, :, 2] = 255 / (max_t - min_t) * (img2[:, :, 2] - min_t)
    return img2

if __name__ == '__main__':
    import os
    video = '002'
    color = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
    root_path = os.path.join("/mnt/data/qzw/data/UCFSports/", 'Frames', video)
    image_list = []
    data_path = "/mnt/data/qzw/data/UCFSports/"
    dataset = "UCFSports"
    train_dataset = tube_dataset.TubeDataset(dataset, data_path=data_path, phase='eval',
                                             modality='rgb',
                                             sequence_length=6)
    height, width = train_dataset.resolution[video]
    s = 10
    target = np.zeros(25)
    target[1:] = train_dataset.gttubes[video][0][0][s: s+6, 1: 5].reshape(-1)
    target[1::2] = target[1::2] / width
    target[2::2] = target[2::2] / height
    for i in range(s, s+6):
        path = os.path.join(root_path, '%06d.jpg' % (i+1))
        image = cv2.imread(path)
        # image = PCA_Jittering(image)
        image = color(image)
        image_list += [image]
    image, target_new = random_crop(np.concatenate(image_list, axis=2), target)
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)
    height_new, width_new, _ = image.shape
    for i in range(6):
        p1 = (int(target_new[i * 4 + 1]*width_new), int(target_new[i * 4 + 2]*height_new))
        p2 = (int(target_new[i * 4 + 3]*width_new), int(target_new[i * 4 + 4]*height_new))
        im1 = (image[:, :, 3*i:3*(i+1)]).astype('uint8')
        cv2.rectangle(im1, p1, p2, (255, 0, 0))
        cv2.imwrite('./image/test{}.jpg'.format(i), im1)
        # image_list += [image]

    # img_new = cv2.imread('./test_img.jpg')
    # print(img_new)
    ss = 0
