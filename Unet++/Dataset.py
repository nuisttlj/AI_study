from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

my_transform = transforms.Compose([transforms.ToTensor()])
data_dir = r"/home/Unet/data"
label_dir = r"/home/Unet/label"


class TumorDataSet(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = ["{0}.png".format(i) for i in range(6247)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_name = os.path.join(data_dir, self.dataset[index])
        label_name = os.path.join(label_dir, self.dataset[index])
        img = Image.open(img_name)
        img_data = my_transform(img)
        label = Image.open(label_name)
        label_data = my_transform(label)

        return img_data, label_data


if __name__ == '__main__':
    dataset = TumorDataSet()
    img, label = dataset[0]
    print(img.size())
    print(label.size())
