# -*- coding: utf-8 -*-
"""
Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class myCNN(nn.Module):
  def __init__(self):
    super(myCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

  def forward(self, x):
    return self.conv1(x)


image_path = '/content/test_image_2.jpg'
image = Image.open(image_path).convert('L')  # Convert to grayscale
transform = transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

model = myCNN()
conv_output_1 = model(image_tensor)
conv_output_2 = model(conv_output_1)
conv_output_3 = model(conv_output_2)

plt.figure(figsize=(20, 80))
plt.subplot(1, 4, 1)
plt.imshow(image_tensor.detach().numpy()[0, 0], cmap='gray')
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(conv_output_1.detach().numpy()[0, 0], cmap='gray')
plt.title('Convoluted_1 Image')

plt.subplot(1, 4, 3)
plt.imshow(conv_output_2.detach().numpy()[0, 0], cmap='gray')
plt.title('Convoluted_2 Image')

plt.subplot(1, 4, 4)
plt.imshow(conv_output_3.detach().numpy()[0, 0], cmap='gray')
plt.title('Convoluted_3 Image')
plt.show()







