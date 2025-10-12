import os
import logging
import numpy as np
import cv2
import glob
import torchvision.transforms.functional as TF
import data.torchvision_x_functional as TF_x
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset

logger = logging.getLogger('base')


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class ImageDataset_HDRplus_480p(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        file = open(os.path.join(root, 'train.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root, "middle_480p", set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root, "output_480p", set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "middle_480p", test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root, "output_480p", test_input_files[i][:-1] + ".jpg"))

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        # img_input = np.array(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio = np.random.uniform(0.8, 1.0)
            W, H = img_expert._size
            crop_h = round(H * ratio)
            crop_w = round(W * ratio)
            # crop_h = 320
            # crop_w = 376
            i, j, h, w = transforms.RandomCrop.get_params(img_expert, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h, crop_w, img_input.shape())
            img_expert = TF.crop(img_expert, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_expert = TF.hflip(img_expert)

            a = np.random.uniform(0.6, 1.4)
            img_input = TF_x.adjust_brightness(img_input, a)

            # a = np.random.uniform(0.8,1.2)
            # img_input = TF_x.adjust_saturation(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_expert = TF.to_tensor(img_expert)

        return {"A_input": img_input, "A_expert": img_expert, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_FiveK_480p(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode
        # file = open(os.path.join(root, 'train_input.txt'), 'r')
        file = open(os.path.join(root, 'train_all.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(
                os.path.join(root, "input", "PNG/480p_16bits_XYZ_WB", set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(
                os.path.join(root, "input", "PNG/480p_16bits_XYZ_WB", test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root, "expertC", "JPG/480p", test_input_files[i][:-1] + ".jpg"))

    def __getitem__(self, index):
        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        # img_input = np.array(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        if self.mode == "train":

            ratio = np.random.uniform(0.8, 1.0)
            W, H = img_expert._size
            crop_h = round(H * ratio)
            crop_w = round(W * ratio)
            i, j, h, w = transforms.RandomCrop.get_params(img_expert, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h, crop_w, img_input.shape())
            img_expert = TF.crop(img_expert, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_expert = TF.hflip(img_expert)

            a = np.random.uniform(0.6, 1.4)
            img_input = TF_x.adjust_brightness(img_input, a)

        img_input = TF_x.to_tensor(img_input)
        img_expert = TF.to_tensor(img_expert)

        return {"A_input": img_input, "A_expert": img_expert, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_HDRplus_4K(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        file = open(os.path.join(root, 'train.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root, "input_tif", set1_input_files[i][:-1] + ".tif"))
            self.set1_expert_files.append(os.path.join(root, "expert", set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "input_tif", test_input_files[i][:-1] + ".tif"))
            self.test_expert_files.append(os.path.join(root, "expert", test_input_files[i][:-1] + ".jpg"))

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        # img_input = np.array(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio = np.random.uniform(0.55, 0.65)
            W, H = img_expert._size
            crop_h = round(H * ratio)
            crop_w = round(W * ratio)
            # crop_h = 320
            # crop_w = 376
            i, j, h, w = transforms.RandomCrop.get_params(img_expert, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h, crop_w, img_input.shape())
            img_expert = TF.crop(img_expert, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_expert = TF.hflip(img_expert)

            a = np.random.uniform(0.6, 1.4)
            img_input = TF_x.adjust_brightness(img_input, a)

            # a = np.random.uniform(0.8,1.2)
            # img_input = TF_x.adjust_saturation(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_expert = TF.to_tensor(img_expert)

        return {"A_input": img_input, "A_expert": img_expert, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_FiveK_4K(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode
        file = open(os.path.join(root, 'train.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root, "Input", set1_input_files[i][:-1]))
            self.set1_expert_files.append(os.path.join(root, "ExpertC", set1_input_files[i][:-1]))

        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "Input", test_input_files[i][:-1]))
            self.test_expert_files.append(os.path.join(root, "ExpertC", test_input_files[i][:-1]))

    def __getitem__(self, index):
        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)], -1)[..., ::-1]
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], -1)[..., ::-1]
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        # img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio = np.random.uniform(0.55, 0.65)
            W, H = img_exptC._size
            crop_h = round(H * ratio)
            crop_w = round(W * ratio)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h, crop_w, img_input.shape())
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.6, 1.4)
            img_input = TF_x.adjust_brightness(img_input, a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_expert": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_LOL(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        self.set1_input_files = sorted(glob.glob(os.path.join(root, "train_x", "*.png")))
        self.set1_expert_files = sorted(glob.glob(os.path.join(root, "train_gt", "*.png")))

        self.test_input_files = sorted(glob.glob(os.path.join(root, "test_x", "*.png")))
        self.test_expert_files = sorted(glob.glob(os.path.join(root, "test_gt", "*.png")))

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        # img_input = np.array(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio = np.random.uniform(0.8, 1.0)
            W, H = img_expert._size
            crop_h = round(H * ratio)
            crop_w = round(W * ratio)
            # crop_h = 320
            # crop_w = 376
            i, j, h, w = transforms.RandomCrop.get_params(img_expert, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h, crop_w, img_input.shape())
            img_expert = TF.crop(img_expert, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_expert = TF.hflip(img_expert)

            a = np.random.uniform(0.6, 1.4)
            img_input = TF_x.adjust_brightness(img_input, a)

            # a = np.random.uniform(0.8,1.2)
            # img_input = TF_x.adjust_saturation(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_expert = TF.to_tensor(img_expert)

        return {"A_input": img_input, "A_expert": img_expert, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_FiveK_8bit(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode
        # file = open(os.path.join(root, 'train_input.txt'), 'r')
        file = open(os.path.join(root, 'train_all.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(
                os.path.join(root, "input", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            self.set1_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(
                os.path.join(root, "input", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            self.test_expert_files.append(os.path.join(root, "expertC", "JPG/480p", test_input_files[i][:-1] + ".jpg"))

    def __getitem__(self, index):
        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], -1)[..., ::-1]
            img_expert = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        # img_input = np.array(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        if self.mode == "train":

            ratio = np.random.uniform(0.8, 1.0)
            W, H = img_expert._size
            crop_h = round(H * ratio)
            crop_w = round(W * ratio)
            i, j, h, w = transforms.RandomCrop.get_params(img_expert, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h, crop_w, img_input.shape())
            img_expert = TF.crop(img_expert, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_expert = TF.hflip(img_expert)

            a = np.random.uniform(0.6, 1.4)
            img_input = TF_x.adjust_brightness(img_input, a)

        img_input = TF_x.to_tensor(img_input)
        img_expert = TF.to_tensor(img_expert)

        return {"A_input": img_input, "A_expert": img_expert, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)
