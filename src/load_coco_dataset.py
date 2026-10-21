import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])  # 读取图像
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(data_dir, label_file):
    images = []
    labels = []

    # 读取标签文件
    label_data = pd.read_csv(label_file, header=None, delimiter=' ')

    for index, row in label_data.iterrows():
        img_id = row[0]  # 假设第一列是图像ID
        img_path = os.path.join(data_dir, f"{img_id}.jpg")  # 根据实际文件名构造路径

        images.append(img_path)

        # 获取性别和年龄标签
        gender = row[39]  # 假设第40列是性别 (male/female)
        age = 0 if row[5] == 0 else 1  # 假设第6列代表年龄 (0=少于30, 1=大于等于30)

        labels.append([gender, age])  # 将性别和年龄组合为标签

    return images, labels


def main():
    data_dir = '../data/peta/images'  # 替换为PETA数据集的图像目录
    label_file = '../data/peta/labels/Label.txt'  # 替换为标签文件的路径

    image_paths, labels = load_data(data_dir, label_file)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # 数据增强和转化
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    val_dataset = CustomDataset(X_val, y_val, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 返回数据加载器
    return train_loader, val_loader


if __name__ == "__main__":
    main()
