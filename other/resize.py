import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('lounge.jpg')
size = img.size
print("Size of Original image:", size)

transform = T.Resize(size = (1400,200))
img = transform(img)

plt.imshow(img)
print("Size after resize:", img.size)
plt.show()
