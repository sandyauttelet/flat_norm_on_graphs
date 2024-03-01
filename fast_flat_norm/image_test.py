from image_to_graph import image_to_graph
import numpy as np
from flat_norm import flat_norm

img_path = 'test_img.png'

grid,E = image_to_graph(img_path)

flat_norm(grid,E,lamb=10**-6,neighbors=4)
