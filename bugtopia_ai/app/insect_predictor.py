import torch
from .utils import load_and_resize_image
from ultralytics import YOLO

model_path = 'app/model/bugtopia_ai_model.pt'
model = YOLO(model_path)

model.eval()

def load_class_names(class_file_path='app/model/classes.txt'):
    class_names = {}
    with open(class_file_path, 'r') as file:
        for line in file:
            index, name = line.strip().split(' ', 1)
            class_names[int(index)] = name
    return class_names

class_names = load_class_names() 

async def predict_insect(img_url: str) -> str:
    img = load_and_resize_image(img_url)
    results = model(img)

    predicted_label = results.pred[0, 0].item()
    insect_name = class_names.get(predicted_label, "Unknown Insect")
    return insect_name