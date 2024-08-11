from PIL import Image
import tifffile

import matplotlib.pyplot as plt
import numpy as np

img = tifffile.imread('linear.tif')
img = np.asarray(Image.open('linear.tif'))
print(repr(img))
"""
plt.imshow(img,  cmap='gray')
plt.show()

plt.hist(img.ravel(), 256, (0, 1))
plt.show()
"""
def arcsinh(image, black, stretch):
    result =  ((image - black) * np.arcsinh(image * stretch)) / (image * np.arcsinh(stretch))
    return result

np_matrix = arcsinh((arcsinh((arcsinh(img, 0.003, 1000)), 0.10, 100)), 0.15, 30)

np_matrix = np.where(img < 0.0057, 0, np.where(img < 0.0058, 0.5, 1))

plt.imshow(np_matrix, vmin=0, vmax=1,cmap='gray')
plt.colorbar()
plt.show()
"""
plt.hist(np_matrix.ravel(), 256, (0, 1))
plt.show()
"""
a = np.array([-0.2, -2]) 


b = np.sign(a) * np.power(np.abs(a), 2)

