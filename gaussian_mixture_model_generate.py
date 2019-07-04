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
                size = z[i])
        i_start = i_end
    return samples



sample_size = 100
mix_coeff = [0.25, 0.25, 0.25, 0.25]
mean = [[-1, 0], [0, 1], [1, 0], [0, -1]]
cov =  [[1, 0], [0, 1],[1, 0], [0, 1],[1, 0], [0, 1],[1, 0], [0, 1]]
cov =  [[1, 0], [0, 1],[1, 0], [0, 1],[1, 0], [0, 1],[1, 0], [0, 1]]
samples = gmm_sample(sample_size, mix_coeff, mean, cov)
print(samples)
