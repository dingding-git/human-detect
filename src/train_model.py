from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F
import torch
import numpy as np

def train(model, train_loader, criterion_sex, criterion_age, criterion_hair, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_sex = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        sex_labels, age_labels, hair_labels = labels[:, 1], labels[:, 0], labels[:, 2]

        optimizer.zero_grad()

        # 前向传播
        sex_output, age_output, hair_output = model(images)

        # 计算损失
        loss_sex = criterion_sex(sex_output, sex_labels)
        loss_age = criterion_age(age_output, age_labels.float())
        loss_hair = criterion_hair(hair_output, hair_labels)

        # 总损失
        loss = loss_sex + loss_age + loss_hair
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(sex_output.data, 1)
        correct_sex += (predicted == sex_labels).sum().item()
        total += sex_labels.size(0)

    accuracy_sex = 100 * correct_sex / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy_sex


def evaluate(model, val_loader, criterion_sex, criterion_age, criterion_hair, device):
    model.eval()
    running_loss = 0.0
    correct_sex = 0
    total = 0
    all_age_preds = []
    all_age_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            sex_labels, age_labels, hair_labels = labels[:, 1], labels[:, 0], labels[:, 2]

            # 前向传播
            sex_output, age_output, hair_output = model(images)

            # 计算损失
            loss_sex = criterion_sex(sex_output, sex_labels)
            loss_age = criterion_age(age_output, age_labels.float())
            loss_hair = criterion_hair(hair_output, hair_labels)

            # 总损失
            loss = loss_sex + loss_age + loss_hair
            running_loss += loss.item()

            # 统计
            _, predicted = torch.max(sex_output.data, 1)
            correct_sex += (predicted == sex_labels).sum().item()
            total += sex_labels.size(0)

            # 收集年龄预测结果
            all_age_preds.append(age_output.cpu().numpy())
            all_age_labels.append(age_labels.cpu().numpy())

    accuracy_sex = 100 * correct_sex / total
    avg_loss = running_loss / len(val_loader)

    # 计算平均绝对误差（MAE）对于年龄
    all_age_preds = np.concatenate(all_age_preds, axis=0)
    all_age_labels = np.concatenate(all_age_labels, axis=0)
    mae_age = mean_absolute_error(all_age_labels, all_age_preds)

    return avg_loss, accuracy_sex, mae_age
