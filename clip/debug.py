from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

mode = "openai/clip-vit-large-patch14-336"

model = CLIPModel.from_pretrained(mode)
processor = CLIPProcessor.from_pretrained(mode)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

print(mode)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
print(logits_per_image)
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)
