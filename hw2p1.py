# dimensions = 512 x 512
# the sequence becomes our "samples"
# since its only 2 dims, we will use (x, y) coords.
# for x, we will use phi_2 and y phi_3 (2, 3 are primes)    

def reverse_bits(num, base):
    reversed_bits = ""
    while (num/base > 0):
        reversed_bits += str(num % base)
        num //= base # floor division
    return reversed_bits 

def convert_to_base_10(bits, base):
    current_val = 1/base
    total = 0.0
    for c in bits:
        total += int(c) * current_val
        current_val /= base
    return total


def generate_halton_samples(num_samples):
    x = []
    y = []
    for idx in range(1, num_samples):
        x_val = convert_to_base_10(reverse_bits(idx, 2), 2) * 512
        y_val = convert_to_base_10(reverse_bits(idx, 3), 3) * 512
        x.append(x_val)
        y.append(y_val)
    return x,y


def load_image_into_pixel_array(image_path : str):
    from PIL import Image
    im = Image.open(image_path)
    im = im.convert('RGB')
    im.load()
    return im

def get_color_value(pixels, x, y):
    return pixels.getpixel((int(x), 512-int(y)-1))

from tkinter import Image
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats.qmc as qmc

def generate_hammersley_samples(num_samples):
    # hammersley is just halton but with a linear sweep for the nth dimension
    x = []
    y = []
    for idx in range(1, num_samples):
        x_val = convert_to_base_10(reverse_bits(idx, 2), 2) * 512
        y_val = idx/num_samples * 512
        x.append(x_val)
        y.append(y_val)
    return x,y

def plot_hammersley_sequence(num_samples, radius):
    x,y = generate_hammersley_samples(num_samples)

    # star discrepancy
    space = np.array([[a,b] for a,b in zip(x,y)])
    l_bounds = [0.0, 0.0]
    u_bounds = [512.0, 512.0]
    space = qmc.scale(space, l_bounds, u_bounds, reverse=True) # scale down to unit cube (normalize between 0 and 1)
    print(qmc.discrepancy(space, method='L2-star'))

    # plot
    image = load_image_into_pixel_array('test_image.png')

    C = [(get_color_value(image, a, b)) for a,b in zip(x,y)]
    C = [(c[0]/255.0, c[1]/255.0, c[2]/255.0) for c in C] # normalize between 0 and 1 for RGB values
    C = np.array(C)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c=C, s=radius)

    plt.show()

plot_hammersley_sequence(500, 200.0)

def plot_halton_sequence(num_samples : int, radius : float):
    x,y = generate_halton_samples(num_samples)

    # star discrepancy
    space = np.array([[a,b] for a,b in zip(x,y)])
    l_bounds = [0.0, 0.0]
    u_bounds = [512.0, 512.0]
    space = qmc.scale(space, l_bounds, u_bounds, reverse=True) # scale down to unit cube (normalize between 0 and 1)
    print(qmc.discrepancy(space, method='L2-star'))

    # plot
    image = load_image_into_pixel_array('test_image.png')

    C = [(get_color_value(image, a, b)) for a,b in zip(x,y)]
    C = [(c[0]/255.0, c[1]/255.0, c[2]/255.0) for c in C] # normalize between 0 and 1 for RGB values
    C = np.array(C)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c=C, s=radius)

    plt.show()

### end fig 1 

### begin fig 2, voronoi diagram
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_voronoi(num_samples):
    x,y = generate_halton_samples(num_samples)

    # star discrepancy

    space = np.array([[a,b] for a,b in zip(x,y)])
    l_bounds = [0.0, 0.0]
    u_bounds = [512.0, 512.0]
    space = qmc.scale(space, l_bounds, u_bounds, reverse=True) # scale down to unit cube (normalize between 0 and 1)
    print(qmc.discrepancy(space, method='L2-star'))

    # plot

    points = np.array([(a, b) for a,b in zip(x,y)])

    vor = Voronoi(points)

    import matplotlib.image as mpimg
    from PIL import ImageOps
    img = mpimg.imread('test_image.png')
    fig = voronoi_plot_2d(vor, line_colors='r', show_vertices=False)
    plt.imshow(img, origin='upper')
    plt.show()

plot_halton_sequence(500, 23.0)
plot_voronoi(100)