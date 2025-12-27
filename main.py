import numpy as np
import torch
my_list =  [[1, 2, 3, 4], [6, 7, 8, 9]]
print(my_list)

np1 = np.random.rand(3, 4)
print("This is my numpy array xxxxxx")
print(np1)
print(np1.dtype)


tensor_2d = torch.randn(3, 4)
print("Here's my 2d tensor xxxxxxx")
print(tensor_2d)
print("Here's my tensor 2d type")
print(tensor_2d.dtype)

tensor_3d = torch.zeros(2, 3, 4)
tensor_3d_1 = torch.ones(2, 3, 4)
print("Here's my zero and ones arrays")
print(tensor_3d)
print(tensor_3d_1)

## Create tensor out of numpy array
my_tensor = torch.tensor(np1)
print("Hers's my_tensor converted from a numpy array xxxxxxx")
print(my_tensor)
