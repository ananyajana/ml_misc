# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:31:17 2019

@author: ananya
this code has been take from https://github.com/mahyarkoy/dmgan_release/blob/master/run_dmgan.py
"""
import numpy as np
import matplotlib.pyplot as plt
# generating 10 gaussians





def generate_circle_data(data_size):
    num_comp = 2
    # picks up data_size number of angles between (0 - 2*pi) from uniform distribution
    z = np.random.uniform(0.0, 2*np.pi, data_size)
    # pick up data_size integers from the array np.arange(num_comp) with replacement and probability of 
    # individual elements listed as p
    ch = np.random.choice(num_comp, size = data_size, replace = True, p = [0.5, 0.5])
    
    x1 = np.sin(z) - 2
    y1 = np.cos(z)
    print(x1)
    print(y1)
    
    x2 = np.sin(z) + 2
    y2 = np.cos(z)
    print(x2)
    print(y2)
    # makes an array like (x11, x21), (x12, x22), (x13, x23) etc
    dx = np.c_[x1, x2]
    dy = np.c_[y1, y2]
    
    # from the data_size number of tuples generated for x values and y values,
    # pick randomly either x1 or x2 and y1 or y2 an dcreate a pair
    data = np.c_[dx[np.arange(data_size), ch], dy[np.arange(data_size), ch]]
    return data
    
def generate_line_data(data_size):
    num_lines = 4
    lb = 0.
    ub = 1.
    
    # picks up data_size number of data between lb and ub from uniform distribution
    z = np.random.uniform(lb, ub, data_size)
    # pick up data_size integers from the array np.arange(num_lines) with replacement and probability of 
    # individual elements listed as p
    ch = np.random.choice(num_lines, size = data_size, replace = True, p = [0.25, 0.25, 0.25, 0.25])
    x1 = z * 0.25 + (1 - z)* .75
    y1 = -1. * x1 + 1.
    
    x2 = -x1
    y2 = -1. * x2 - 1.
    
    x3 = x1
    y3 = 1. * x3 - 1.
    
    x4 = -x1
    y4 = 1. * x4 + 1.
    
    dx = np.c_[x1, x2, x3, x4]
    dy = np.c_[y1, y2, y3, y4]
    
    data = np.c_[dx[np.arange(data_size), ch], dy[np.arange(data_size), ch]]
    return data
    
#def plot_dataset(datasets, color, pathname, title = 'Dataset', fov = 2):
def plot_dataset(dataset):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.clear()
    
    plt.scatter(dataset[:, 0], dataset[:, 1])
    
datasets = generate_circle_data(500)
datasets2 = generate_line_data(500)
print(datasets)
#plot_dataset(datasets)
    
print(datasets2)
plot_dataset(datasets2)

