import torch

# 모델 파일 경로
model_path = './model/bugtopia_ai_model.pt'

# 모델 로드
model = torch.load(model_path)

print(model)