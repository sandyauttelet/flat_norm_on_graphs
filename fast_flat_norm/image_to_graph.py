import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'test_img.png'

def primitive_threshold(image,black_white_tolerance):
    if image < black_white_tolerance:
        return 255
    else:
        return 0 #0 is black
    
def scale(x,y):
    return x/y

def img_to_cartesian(x,y,h):
    x,y = y, h-x
    return x,y

def image_to_graph(img_path,black_white_tolerance=255//2):
    threshold = np.vectorize(primitive_threshold)
    img = cv2.imread(img_path, 0)
    image_height = np.shape(img)[0]
    img = threshold(img,black_white_tolerance)
    grid = np.indices(np.shape(img))
    max_x = np.max(grid[0])
    max_y = np.max(grid[1])
    grid = img_to_cartesian(grid[0],grid[1],image_height)
    grid_x_indices = scale(grid[0],max_x)
    grid_y_indices = scale(grid[1],max_y)
    
    #plt.figure(0)
    #plt.imshow(img)
    #plt.figure(1)
    #plt.scatter(grid_x_indices, grid_y_indices)
    #plt.scatter(img_x_indices, img_y_indices)
    return np.array(grid2graph(grid_x_indices,grid_y_indices)),img.astype(bool).flatten()

def grid2graph(xs,ys):
    g = []
    for i in range(len(xs[0])):
        g += list(zip(xs[i],ys[i]))
    assert len(g) == xs.shape[0]*xs.shape[1]
    return g

def plot_graph_as_image(G):
    xs,ys = [],[]
    for x in G.nodes:
        xs.append(x[0])
        ys.append(x[1])
    #xs,ys = img_to_cartesian(np.array(xs), np.array(ys), len(xs))
    #print(xs)
    #print(ys)
    plt.figure()
    plt.scatter(xs,ys)

# p = [(i,j) for i in range(3) for j in range(3)]
# xs = [x[0] for x in p] 
# ys = [x[1] for x in p]
# print(p)
# print(xs)
# #plt.scatter(xs,ys)

if __name__ == "__main__":
    # TODO: there is some bug with non square images
    grid,E = image_to_graph(img_path)
    print(grid.shape)
    print(E)
    plt.show()
    #plot_graph_as_image(grid)
