import numpy as np
import cv2
from subpixel_edges import subpixel_edges


image1 = cv2.imread(r"examples/images/circle2.png")
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.float64)


edges, grad, absGxInner, absGyInner = subpixel_edges(image1_gray, 25, 2, 2)
print(edges.x.shape, edges.y.shape)
print("pixel_x: ", edges.pixel_x)
print("pixel_y: ", edges.pixel_y)
print("subpixel_x: ", edges.x)
print("subpixel_y: ", edges.y)
print("nx: ", edges.nx)
print("ny: ", edges.ny)
print("curv: ", edges.curv)
print("i0: ", edges.i0)
print("i1: ", edges.i1)
