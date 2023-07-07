# Authors: DeeKay Goswami & Naresh Kumar Devulapally

import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import matplotlib.pyplot as plt

segmentation_model = deeplabv3_resnet50(pretrained=True)

segmentation_model.eval()
image_path = 'utils/images/test_segment2.png'
image = Image.open(image_path).convert("RGB")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

segmentation_output = segmentation_model(image_tensor)['out']
segmentation_map = torch.argmax(segmentation_output.squeeze(), dim=0).detach().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image)
axes[0].axis('off')
axes[0].set_title('Original Image')
axes[1].imshow(segmentation_map)
axes[1].axis('off')
axes[1].set_title('Car Segmentation')

plt.tight_layout()
plt.show()
