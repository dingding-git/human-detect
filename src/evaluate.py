import torch
from model import AttributeModel
from data_loader import PETADataLoader
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据加载器
test_dataset = PETADataLoader(data_dir='data', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
model = AttributeModel().to(device)
model.load_state_dict(torch.load('attribute_model.pth'))
model.eval()

correct_gender = 0
correct_age = 0
correct_hair = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        gender_labels, age_labels, hair_labels = labels[:, 0], labels[:, 1], labels[:, 2]

        gender_output, age_output, hair_output = model(inputs)

        _, predicted_gender = torch.max(gender_output, 1)
        _, predicted_age = torch.max(age_output, 1)
        _, predicted_hair = torch.max(hair_output, 1)

        correct_gender += (predicted_gender == gender_labels).sum().item()
        correct_age += (predicted_age == age_labels).sum().item()
        correct_hair += (predicted_hair == hair_labels).sum().item()
        total += gender_labels.size(0)

accuracy_gender = correct_gender / total
accuracy_age = correct_age / total
accuracy_hair = correct_hair / total

print(f'Gender Accuracy: {accuracy_gender:.4f}')
print(f'Age Accuracy: {accuracy_age:.4f}')
print(f'Hair Accuracy: {accuracy_hair:.4f}')
