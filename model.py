import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load the pretrained model
model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Load ImageNet Class Labels
imagenet_classes = []
with open("imagenet_classes.txt") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

#Define the image transformations for the image
transform = transforms.Compose([            #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(size=224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225]                  #[7]
    )])

def predict(image_paths):
    """Process multiple images and return predictions."""
    images = [transform(Image.open(img_path).convert("RGB")) for img_path in image_paths]
    batch = torch.stack(images)  # Convert list to batch tensor

    with torch.no_grad():
        outputs = model(batch)
    
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    class_indices = torch.argmax(probabilities, dim=1)

    results = []
    for img_path, class_index, prob in zip(image_paths, class_indices, probabilities):
        results.append({
            "filename": os.path.basename(img_path),
            "class": imagenet_classes[int(class_index.item())],
            "confidence": f"{prob[class_index].item():.2%}",
        })

    return results

if __name__ == '__main__':
    result = predict(['data/black_swan.jpg'])
    print(result)
    

    