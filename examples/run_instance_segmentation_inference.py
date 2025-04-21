import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# Load image
image = Image.open(
    requests.get(
        "http://farm4.staticflickr.com/3017/3071497290_31f0393363_z.jpg", stream=True
    ).raw
)

# Load model and image processor
device = "cuda"
checkpoint = "qubvel-hf/finetune-instance-segmentation-ade20k-mini-mask2former"

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    checkpoint, device_map=device
)
image_processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)

# Run inference on image
inputs = image_processor(images=[image], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Post-process outputs
outputs = image_processor.post_process_instance_segmentation(
    outputs, target_sizes=[(image.height, image.width)]
)

print("Mask shape: ", outputs[0]["segmentation"].shape)
print("Mask values: ", outputs[0]["segmentation"].unique())
for segment in outputs[0]["segments_info"]:
    print("Segment: ", segment)

segmentation = outputs[0]["segmentation"].numpy()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.array(image))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(segmentation)
plt.axis("off")
plt.savefig("examples/outputs/instance_segmentation.png")
