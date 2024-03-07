from image_to_graph import image_to_graph
import numpy as np
from flat_norm import flat_norm
import matplotlib.pyplot as plt
from time import perf_counter

if __name__ == "__main__":
    img_path = 'circles2.png'

    grid,E = image_to_graph(img_path)
    tick = perf_counter()
    fn, _, _, _ = flat_norm(grid,E,lamb=0.002,neighbors=4)
    tock = perf_counter()
    print(tock-tick)
    plt.figure(0)
    plt.scatter(grid[E][:, 0], grid[E][:, 1], color="black")
    plt.axis("off")
    plt.figure(1)
    plt.scatter(fn[:,0],fn[:,1],color="royalblue")
    plt.axis("off")
    plt.show()

