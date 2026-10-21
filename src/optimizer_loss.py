import torch.optim as optim
import torch.nn as nn

def get_loss_fn():
    # 性别预测使用交叉熵损失，年龄使用MSE损失，头发长短使用交叉熵损失
    sex_loss_fn = nn.CrossEntropyLoss()
    age_loss_fn = nn.MSELoss()
    hair_loss_fn = nn.CrossEntropyLoss()
    return sex_loss_fn, age_loss_fn, hair_loss_fn

def get_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)
