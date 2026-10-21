import torch
import torchvision.transforms as transforms
from PIL import Image

class AttributeRecognition:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.model.eval()  # 设置为评估模式
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def load_model(self, model_path):
        model = AttributeRecognitionModel()  # 确保模型定义在此处
        model.load_state_dict(torch.load(model_path))
        return model

    def predict(self, image):
        image = self.transform(image).unsqueeze(0)  # 增加批次维度
        with torch.no_grad():
            output = self.model(image.float())
            gender = 'male' if output.item() > 0.5 else 'female'  # 根据输出判断性别
            return {'gender': gender}
