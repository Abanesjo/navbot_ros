import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

# --- Map metadata (from yaml) ---
resolution      = 0.025         # metres per pixel
origin          = [-6.1, -3.69] # world coords (metres) of bottom-left pixel of full image
occupied_thresh = 0.65
free_thresh     = 0.25

# --- Load map ---
map_arr = cv2.imread('map1/map_processed.png', cv2.IMREAD_GRAYSCALE)
height, width = map_arr.shape

#coordinates in pixel space
def world_to_pixel(x, y):
    col = int(np.floor((x - origin[0]) / resolution))
    row = int(np.floor(height - (y - origin[1]) / resolution - 1))

    if col < 0:
        col = 0
    if col >= width:
        col = width-1
    if row < 0:
        row = 0
    if row >= height:
        row = height-1
        
    return col, row

occ_pixel_thresh  = int((1.0 - occupied_thresh) * 255)  # ~89

obstacle_mask = map_arr <= occ_pixel_thresh

distance_field = distance_transform_edt(~obstacle_mask) * resolution

#Example
#Suppose we have a LiDAR ray that lands at the point (2, 1). We calculate the distance to the nearest goal.
x_query = 2.0
y_query = 1.0

col_query, row_query = world_to_pixel(x_query, y_query)
distance = distance_field[row_query, col_query]
print(f"At {x_query} and {y_query} the distance is {distance}")

#--------------Plotting--------------------_#
fig, ax = plt.subplots()
im = ax.imshow(distance_field, cmap='plasma', origin='upper')
plt.colorbar(im, ax=ax, label='Distance (m)')

# Origin
ox, oy = world_to_pixel(0.0, 0.0)
ax.plot(ox, oy, 'g+', markersize=10, markeredgewidth=2, label='origin')

# Query point
ax.plot(col_query, row_query, 'r+', markersize=10, markeredgewidth=2,
        label=f'query ({x_query}, {y_query}) d={distance:.2f}m')

ax.legend()
plt.tight_layout()
plt.savefig('distance_field.png', dpi=150)