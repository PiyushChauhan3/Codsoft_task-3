import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer

resnet_weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=resnet_weights)
resnet = resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
   
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # type: ignore 
    return image



model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def generate_caption(image_tensor):
   
    output_ids = model.generate(image_tensor, max_length=16, num_beams=4)  # type: ignore 
    attention_mask = torch.ones_like(output_ids)  # type: ignore 
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)  
    return caption


def image_captioning(image_path):
   
    image_tensor = preprocess_image(image_path)
    caption = generate_caption(image_tensor)
    return caption

if __name__ == "__main__":
    image_path = "img.jpg"  
    caption = image_captioning(image_path)
    print("Generated Caption:", caption)
