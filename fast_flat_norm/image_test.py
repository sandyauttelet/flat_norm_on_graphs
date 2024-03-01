from image_to_graph import image_to_graph
import numpy as np
from flat_norm import flat_norm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_path = 'spike.png'

    grid,E = image_to_graph(img_path)
    fn, _, _, _ = flat_norm(grid,E,lamb=0.01,neighbors=8)
    plt.scatter(grid[E][:,0], grid[E][:,1],color="b")
    plt.scatter(fn[:,0],fn[:,1],color="r")
    plt.show()

