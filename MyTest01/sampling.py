from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        # self.dataset.extend(open(os.path.join(path, "negative1.txt")).readlines())

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split()
        img_path = os.path.join(self.path, strs[0])
        cond = torch.Tensor([float(strs[1])])
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        used_label = torch.Tensor([int(strs[6])])
        # img_data = Image.open(img_path)
        # img_data = img_data.convert('RGB')
        # img_data = torch.Tensor(np.array(img_data) / 255. - 0.5)
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5)
        img_data = img_data.permute(2, 0, 1)
        return img_data, cond, offset, used_label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = FaceDataset(r"E:\myceleba\12")
    print(dataset[0])
