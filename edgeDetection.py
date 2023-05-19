from PIL import ImageFilter
from PIL import Image

with Image.open("Bird_Demo.jpg") as im:
    im.filter(ImageFilter.FIND_EDGES).show()

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

# Apply guassian filter to the image to reduce noise
# Then Sobel filter to get the gradient in x and y direction
# Then go thru the pixels and categoirze them as 