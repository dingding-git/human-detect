import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm  # 导入 tqdm 库


class PA100KDataset(Dataset):
    def __init__(self, image_file, label_file, data_dir, transform=None):
        # 读取图像文件路径和标签
        with open(image_file, 'r') as f:
            self.image_names = f.readlines()
        with open(label_file, 'r') as f:
            self.labels = f.readlines()

        # 确保图像路径和标签数量匹配
        assert len(self.image_names) == len(self.labels), "Image and label files do not match in length."

        self.data_dir = data_dir  # 图像文件夹的根目录
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 获取图像路径，并去除换行符，并确保没有额外的符号
        img_name = self.image_names[idx].strip().strip("[]").replace("'", "")
        img_name = os.path.join(self.data_dir, img_name)

        # 解析标签
        label = self.labels[idx].strip().split(',')  # 分割为列表
        gender, age = int(label[0]), int(label[1])  # 提取性别和年龄标签

        # 加载图像
        image = Image.open(img_name)

        # 应用数据增强和预处理
        if self.transform:
            image = self.transform(image)

        return image, gender, age


# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. 定义模型
class AttributeRecognitionModel(nn.Module):
    def __init__(self, num_classes_gender=2, num_classes_age=3):
        super(AttributeRecognitionModel, self).__init__()

        # 使用预训练的 ResNet50 模型
        self.resnet = models.resnet50(pretrained=True)

        # 替换最后的全连接层
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes_gender + num_classes_age)  # 性别 + 年龄
        )

    def forward(self, x):
        return self.resnet(x)


# 4. 训练和验证函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个 epoch 包括训练和验证阶段
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects_gender = 0
            running_corrects_age = 0
            dataset_size = 0

            # 使用 tqdm 显示进度条
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}', dynamic_ncols=True)

            for inputs, gender_labels, age_labels in progress_bar:
                inputs = inputs.cuda()
                gender_labels = gender_labels.cuda()
                age_labels = age_labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    gender_outputs, age_outputs = outputs[:, :2], outputs[:, 2:]

                    # 计算性别和年龄的损失
                    loss_gender = criterion[0](gender_outputs, gender_labels)
                    loss_age = criterion[1](age_outputs, age_labels)
                    loss = loss_gender + loss_age

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                    # 计算准确率
                    preds_gender = gender_outputs.argmax(dim=1)
                    preds_age = age_outputs.argmax(dim=1)
                    running_corrects_gender += (preds_gender == gender_labels).sum().item()
                    running_corrects_age += (preds_age == age_labels).sum().item()
                    dataset_size += inputs.size(0)

                    # 更新进度条
                    progress_bar.set_postfix(loss=running_loss / dataset_size,
                                             acc_gender=running_corrects_gender / dataset_size,
                                             acc_age=running_corrects_age / dataset_size)

            epoch_loss = running_loss / dataset_size
            epoch_acc_gender = running_corrects_gender / dataset_size
            epoch_acc_age = running_corrects_age / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc_gender: {epoch_acc_gender:.4f} Acc_age: {epoch_acc_age:.4f}')

    return model


# 5. 主训练代码
if __name__ == '__main__':
    # 准备数据
    data_dir = '../data/PA100K/data/release_data/release_data'  # 图像文件夹的路径
    train_dataset = PA100KDataset(image_file='../data/PA100K/annotation/train_images_name.txt', label_file='../data/PA100K/annotation/train_label.txt', data_dir=data_dir,
                                  transform=transform)
    val_dataset = PA100KDataset(image_file='../data/PA100K/annotation/val_images_name.txt', label_file='../data/PA100K/annotation/val_label.txt', data_dir=data_dir,
                                transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    dataloaders = {'train': train_loader, 'val': val_loader}

    # 初始化模型
    model = AttributeRecognitionModel().cuda()

    # 定义损失函数和优化器
    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()
    criterion = [criterion_gender, criterion_age]
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

    # 保存模型
    torch.save(model.state_dict(), '../models/attribute_recognition_model.pth')
    print("Model saved as 'attribute_recognition_model.pth'")
