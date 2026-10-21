import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class PetaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # 遍历各个类别文件夹并根据文件夹名称创建标签
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(os.path.join(root, file))

                    # 从文件夹名提取标签
                    folder_name = os.path.basename(root)
                    sex_label = 0 if 'Female' in folder_name else 1  # 假设文件夹名包含 "Female" 或 "Male"
                    if 'Age16_30' in folder_name:
                        age_label = 16
                    elif 'Age31_45' in folder_name:
                        age_label = 31
                    elif 'Age46_60' in folder_name:
                        age_label = 46
                    else:
                        age_label = 61  # 假设所有大于60的年龄都归为此类

                    hair_label = 0 if 'hairShort' in folder_name else 1  # 假设文件夹名包含 "hairShort" 或 "hairLong"

                    self.labels.append((age_label, sex_label, hair_label))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        # 获取标签
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 数据增强（例如：随机水平翻转、旋转、裁剪等）
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = PetaDataset(image_dir='../data/from_PETA', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
