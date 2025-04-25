import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

SKIN_PALETTE = [
    # Skin palette that maps each class to RGB values.
    [220, 220, 220],  # background
    [120, 120, 120],  # skin
]

# Load image
image = Image.open("examples/inputs/acne.jpg")

# Load model and image processor
device = "cuda"
checkpoint = "checkpoints/segformer-b5-acne-reduce-labels"

model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, device_map=device)
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# Run inference on image
inputs = image_processor(images=[image], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_segmentation_map = image_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]
predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
print(predicted_segmentation_map)

color_seg = np.zeros(
    (predicted_segmentation_map.shape[0], predicted_segmentation_map.shape[1], 3),
    dtype=np.uint8,
)  # height, width, 3

palette = np.array(SKIN_PALETTE)
for label, color in enumerate(palette):
    color_seg[predicted_segmentation_map == label, :] = color
# Convert to BGR
color_seg = color_seg[..., ::-1]

# Show image + mask
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.savefig("examples/outputs/acne_segmentation_B5.png")
