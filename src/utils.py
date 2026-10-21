import os
from torch.utils.data import Dataset
from PIL import Image
import torch


class PedestrianDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels_gender = []
        self.labels_age = []
        self.labels_hair = []

        # 遍历目录获取所有图片路径及标签
        for gender_folder in os.listdir(root_dir):
            gender_folder_path = os.path.join(root_dir, gender_folder)
            if os.path.isdir(gender_folder_path):
                # 解析性别标签
                gender_label = self._get_gender_label(gender_folder)
                for img_name in os.listdir(gender_folder_path):
                    img_path = os.path.join(gender_folder_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels_gender.append(gender_label)
                    self.labels_age.append(-1)  # 默认的年龄标签
                    self.labels_hair.append(-1)  # 默认的发型标签

        # 遍历剩余的年龄和发型文件夹，更新相应的标签
        for age_folder in os.listdir(root_dir):
            age_folder_path = os.path.join(root_dir, age_folder)
            if os.path.isdir(age_folder_path):
                age_label = self._get_age_label(age_folder)
                for img_name in os.listdir(age_folder_path):
                    img_path = os.path.join(age_folder_path, img_name)
                    # 根据年龄文件夹名来更新对应的标签
                    for idx, path in enumerate(self.image_paths):
                        if path == img_path:
                            self.labels_age[idx] = age_label

        for hair_folder in os.listdir(root_dir):
            hair_folder_path = os.path.join(root_dir, hair_folder)
            if os.path.isdir(hair_folder_path):
                hair_label = self._get_hair_label(hair_folder)
                for img_name in os.listdir(hair_folder_path):
                    img_path = os.path.join(hair_folder_path, img_name)
                    # 根据发型文件夹名来更新对应的标签
                    for idx, path in enumerate(self.image_paths):
                        if path == img_path:
                            self.labels_hair[idx] = hair_label

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        gender_label = self.labels_gender[idx]
        age_label = self.labels_age[idx]
        hair_label = self.labels_hair[idx]

        # 数据增强
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([gender_label, age_label, hair_label], dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def _get_gender_label(self, folder_name):
        """
        根据文件夹名称提取性别标签
        :param folder_name: 文件夹名称，格式为 '0_Female' 或 '1_Male'
        :return: 性别标签（0 或 1）
        """
        if 'Female' in folder_name:
            return 0
        elif 'Male' in folder_name:
            return 1
        return -1  # 无效标签

    def _get_age_label(self, folder_name):
        """
        根据文件夹名称提取年龄标签
        :param folder_name: 文件夹名称，格式为 'Age16_30', 'Age31_45', 等
        :return: 年龄标签（0、1、2、3）
        """
        if 'Age16_30' in folder_name:
            return 0
        elif 'Age31_45' in folder_name:
            return 1
        elif 'Age46_60' in folder_name:
            return 2
        elif 'AgeAbove60' in folder_name:
            return 3
        return -1  # 无效标签

    def _get_hair_label(self, folder_name):
        """
        根据文件夹名称提取发型标签
        :param folder_name: 文件夹名称，格式为 'hairLong' 或 'hairShort'
        :return: 发型标签（0 或 1）
        """
        if 'hairLong' in folder_name:
            return 0
        elif 'hairShort' in folder_name:
            return 1
        return -1  # 无效标签
