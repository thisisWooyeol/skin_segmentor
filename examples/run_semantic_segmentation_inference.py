import os
import resource

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

SKIN_PALETTE = [
    # Skin palette that maps each class to RGB values.
    [120, 120, 120],  # background
    [220, 220, 220],  # skin
]

# Profiling setup
process = psutil.Process(os.getpid())
print(f"Memory before loading model: {process.memory_info().rss / (1024**2):.2f} MB")

# Use CPU for inference
device = "cpu"
checkpoint = "checkpoints/segformer-b5-acne-reduce-labels-focal+dice"

# Load model and processor
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint).to(device)
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
print(f"Memory after loading model: {process.memory_info().rss / (1024**2):.2f} MB")

# Load and preprocess image
image = Image.open("examples/inputs/acne.jpg")
inputs = image_processor(images=[image], return_tensors="pt").to(device)
print(f"Memory after preprocessing: {process.memory_info().rss / (1024**2):.2f} MB")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
print(f"Memory after inference: {process.memory_info().rss / (1024**2):.2f} MB")

# Report peak RAM usage (ru_maxrss is in KB on Linux)
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak_kb / 1024
print(f"Peak memory usage: {peak_mb:.2f} MB")

# Post-process segmentation map
predicted_segmentation_map = image_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]
predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

# Colorize segmentation
palette = np.array(SKIN_PALETTE)
color_seg = np.zeros(
    (predicted_segmentation_map.shape[0], predicted_segmentation_map.shape[1], 3),
    dtype=np.uint8,
)
for label, color in enumerate(palette):
    color_seg[predicted_segmentation_map == label, :] = color
# Convert to BGR for OpenCV compatibility
color_seg = color_seg[..., ::-1]

# Blend original and mask for visualization
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

# Save and display
plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.axis("off")
plt.savefig("examples/outputs/acne_segmentation_B5.png")
