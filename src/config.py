import torch

# 数据集配置
DATASET_PATH = '../data/from_PETA'
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别标签
GENDER_CLASSES = ['Female', 'Male']
AGE_CLASSES = ['Age16_30', 'Age31_45', 'Age46_60', 'AgeAbove60']
HAIR_CLASSES = ['HairLong', 'HairShort']

# 模型保存路径
MODEL_SAVE_PATH = '../models/attribute_model.pth'
