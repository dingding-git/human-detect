import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class PedestrianAttributeModel(nn.Module):
    def __init__(self, num_classes_age=1, num_classes_sex=2, num_classes_hair=2):
        super(PedestrianAttributeModel, self).__init__()

        # 使用预训练的ResNet-18作为特征提取网络
        self.backbone = models.resnet18(pretrained=True)
        # 修改最后的全连接层以适应我们的任务
        self.backbone.fc = nn.Identity()  # 移除原有分类层

        # 性别分类
        self.fc_sex = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_sex)  # 0: Male, 1: Female
        )

        # 年龄回归
        self.fc_age = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_age)  # 年龄预测
        )

        # 头发长短分类
        self.fc_hair = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_hair)  # 0: Short hair, 1: Long hair
        )

    def forward(self, x):
        features = self.backbone(x)

        # 性别、年龄、头发长短预测
        sex_output = self.fc_sex(features)
        age_output = self.fc_age(features)
        hair_output = self.fc_hair(features)

        return sex_output, age_output, hair_output
