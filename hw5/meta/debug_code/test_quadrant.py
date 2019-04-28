import math
import numpy as np

delta = 4
is_evaluation = False

num_groups = math.ceil(21./delta)
even_num_groups = math.ceil(float(num_groups)/2)
odd_num_groups = math.floor(float(num_groups)/2)

a = np.zeros([21,21])

for _ in range(1000):
    if is_evaluation == False:
        x_group = np.random.randint(0, num_groups)    
        if x_group % 2 == 0: 
            y_group = 2*np.random.randint(0, even_num_groups) 
        else:
            y_group = 2*np.random.randint(0, odd_num_groups) + 1
    else:
        x_group = np.random.randint(0, num_groups)    
        if x_group % 2 == 0: 
            y_group = 2*np.random.randint(0, odd_num_groups) + 1
        else:
            y_group = 2*np.random.randint(0, even_num_groups)

    delta_x = np.random.uniform(0, delta)
    delta_y = np.random.uniform(0, delta)

    x = x_group * delta + delta_x
    y = y_group * delta + delta_y

    x = x if x < 20 else 20
    y = y if y < 20 else 20

    a[math.floor(x), math.floor(y)] += 1

print((a>0).astype(int))


