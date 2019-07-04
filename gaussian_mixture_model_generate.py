#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:14:58 2019

@author: ananya
"""
import numpy as np
import matplotlib.pyplot as plt


def gmm_sample(num_samples, mix_coeffs, mean, cov):
    # draws samples from multinomial distributions according to the probability distribution mix_coeffs
    z = np.random.multinomial(num_samples, mix_coeffs)
    samples = np.zeros(shape = [num_samples, len(mean[0])])
    i_start = 0
    for i in range(len(mix_coeffs)):
        i_end = i_start + z[i]
        samples[i_start:i_end, :] = np.random.multivariate_normal(
                mean = np.array(mean)[i, :],
                cov = np.diag(np.array(cov)[i, :]),
                #cov = np.array(cov)[i, :],
                size = z[i])
        i_start = i_end
        plt.scatter(samples[:, 0], samples[:, 1])
    return samples



sample_size = 100
mix_coeff = [0.25, 0.25, 0.25, 0.25]
mean = [[-5, 0], [0, 5], [5, 0], [0, -5]]
cov =  [[0.2, 0.2], [0.2, 0.2],[0.2, 0.2], [0.2, 0.2],[0.2, 0.2], [0.2, 0.2],[0.2, 0.2], [0.2, 0.2]]
samples = gmm_sample(sample_size, mix_coeff, mean, cov)
print(samples)
#print(samples[:, 0])
#print(samples[:, 1])
plt.scatter(samples[:, 0], samples[:, 1])

