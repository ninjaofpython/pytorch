import torch
import numpy as np

# Basic tensor math operations

tensor_a = torch.tensor([1, 2, 3, 4])
tensor_b = torch.tensor([5, 6, 7, 8])


# Addition
print("Adding tensor_a and tensor_b with actual arithmetic xxxxxxxxx.")
print(tensor_a + tensor_b)


# Addition Longhand result same as above
print("Addition 'Longhand' but clean syntax xxxxxx.")
print(torch.add(tensor_a, tensor_b))


# Subtraction
print("Here's subtraction with Pytorch xxxxxx.")
print(tensor_b - tensor_a)

