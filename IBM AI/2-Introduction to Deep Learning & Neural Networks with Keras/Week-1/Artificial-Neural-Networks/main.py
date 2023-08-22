import numpy as np # Numpy kütüphanesini kullanalım 
from random import seed

# weights = np.around(np.random.uniform(size=6), decimals=2) # weights değerleri
# biases = np.around(np.random.uniform(size=3), decimals=2) # bias değerleri

# x_1 = 0.5 # input 1
# x_2 = 0.85 # input 2

# z_11 = ( x_1 * weights[0] ) + ( x_2 * weights[1] ) + biases[0]
# z_12 = ( x_1 * weights[2] ) + ( x_2 * weights[3] ) + biases[1]

# a_11 = 1.0 / (1.0 + np.exp(-z_11))
# a_12 = 1.0 / (1.0 + np.exp(-z_12))

# z_2 = (a_11 * weights[4]) + (a_12 * weights[5]) + biases[2]

# a_2 = 1.0 / (1.0 + np.exp(-z_2))

# print(a_2)

# def compute_weighted_sum(inputs, weights, bias):
#     return np.sum(inputs * weights) + bias


# np.random.seed(12)
# inputs = np.around(np.random.uniform(size=5), decimals=2)

# # print('The inputs to the network are {}'.format(inputs))

# compute_weighted_sum()

a_12 = 1.0 / (1.0 + np.exp(-0.267))
print(a_12)