import os
import cv2
from tqdm import tqdm
import torch


class ISIC17Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = "./data/isic2017"
        image_dir = os.path.join(self.data_path, "ISIC-2017_Training_Data")
        self.data = [f"{image_dir}/{file}" for file in os.listdir(image_dir) if file.endswith(".jpg")]
        image_dir = os.path.join(self.data_path, "ISIC-2017_Test_v2_Data")
        self.data += [f"{image_dir}/{file}" for file in os.listdir(image_dir) if file.endswith(".jpg")]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = self.data[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        n_pixel = img.shape[0]*img.shape[1]

        img = torch.tensor(img, dtype=torch.float64)
        n_pixel = torch.tensor(n_pixel, dtype=torch.float64)
        return img, n_pixel


class ISIC18Dataset(ISIC17Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = "./data/isic2018"
        image_dir = os.path.join(self.data_path, "ISIC2018_Task1-2_Training_Input")
        self.data = [f"{image_dir}/{file}" for file in os.listdir(image_dir) if file.endswith(".jpg")]
        image_dir = os.path.join(self.data_path, "ISIC2018_Task1-2_Test_Input")
        self.data += [f"{image_dir}/{file}" for file in os.listdir(image_dir) if file.endswith(".jpg")]


@torch.no_grad()
def cal_mean_and_std_fast(dataset):
    sum = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device="cuda:0")
    sum_sq = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device="cuda:0")
    all_pixels = torch.tensor(0, dtype=torch.float64, device="cuda:0")

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    for images, n_pixels in tqdm(loader):
        images = images.to("cuda:0").view(-1, 3)/255.0
        sum += images.sum(dim=0)
        sum_sq += (images ** 2).sum(dim=0)
        all_pixels += n_pixels.sum().to("cuda:0")

    # mean and std
    total_mean = sum / all_pixels
    total_var = (sum_sq / all_pixels) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print(f"Dataset MEAN is: {total_mean}")
    print(f"Dataset STD is: {total_std}")


if __name__ == '__main__':
    # dataset = ISIC17Dataset()
    dataset = ISIC18Dataset()
    cal_mean_and_std_fast(dataset)