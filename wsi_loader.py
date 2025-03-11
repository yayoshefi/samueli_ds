import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from python_wsi_preproc import slide

class WSIDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx].filename
        label = self.data.iloc[idx].tag
        slide_num = self.data.iloc[idx].slide_num
        
        slide_path = os.path.join(self.data_dir, img_name)
        img_path = slide.get_training_image_path(slide_num)

        try:
            # TODO: read slide and extract image
            # TODO": how to treat the test images? I would need to acutally run on the entire slide not only crop
            image = Image.open(img_path).convert('RGB')
            # Placeholder for image cropping functionality
            # image = crop_image(image)
        except Exception as e:
            print(f"failed in wsi {slide_num} : {slide_path=}, {img_path=}")
            raise e

        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return image, label

def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    target = torch.tensor([item[1] for item in batch]).reshape(-1, 1)
    return [data, target]

# Transforms
# ===========
train_transform = A.Compose([
    A.Resize(224, 224),
    A.RandomCrop(200, 200),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Build DataLoader
# ================
def create_dataloader(csv_file, data_dir, batch_size=32, num_workers=4, is_train=True):
    transform=train_transform if is_train else test_transform
    dataset = WSIDataset(csv_file, data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if is_train else False,
        collate_fn=collate_fn
        )
    return dataloader


if __name__ == "__main__":
    # Test the dataloader
    loader = create_dataloader('data/datasets/tupac_16_100/data.csv', 'data/raw/training_png', batch_size=32, num_workers=0, is_train=True)
    for x, y in loader:
        print(x.shape, y.shape)
        break